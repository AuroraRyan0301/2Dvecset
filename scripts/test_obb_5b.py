"""REAL integration: load Wan2.2-TI2V-5B weights into WanModel_OBB, run forward + train steps (GPU).

On the actual pretrained backbone (trained head -> gradients flow): OBB cross-attn injects, backbone
frozen, OBB branch (injectors + embedder) trains, real velocity output shape correct. Flow loss only
(self-tracking added later once features are exposed).

Run:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD <trellis2-python> scripts/test_obb_5b.py
"""
import glob
import math
import os

import torch
from safetensors.torch import load_file

from wan.modules.obb import WanModel_OBB, random_scene, build_obb_tokens, build_plucker
from wan.modules.obb.losses import flow_matching_loss

CKPT = "/gs/fs/tga-koike-shanda4/yurh/2Dvecset/_refsrc/ckpts/Wan2.2-TI2V-5B"


def main():
    torch.manual_seed(0)
    dev = "cuda"
    F, H, W = 4, 16, 16                                   # latent; token grid (4,8,8), S=256
    lat_hw, S, D_APP = (H // 2, W // 2), F * (H // 2) * (W // 2), 32

    print("building WanModel_OBB (TI2V-5B backbone) ...")
    model = WanModel_OBB(
        model_type='ti2v', patch_size=(1, 2, 2), text_len=512, in_dim=48, dim=3072, ffn_dim=14336,
        freq_dim=256, text_dim=4096, out_dim=48, num_heads=24, num_layers=30,
        obb_inject_layers=[0, 10, 20],
        obb_d_app=D_APP)
    sd = {}
    for shard in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
        sd.update(load_file(shard))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded backbone; missing={len(missing)} (new OBB params) unexpected={len(unexpected)}; "
          f"head trained nonzero={float(model.head.head.weight.abs().sum())>0}")

    # bf16 to fit ~34GB: cast only params (the complex RoPE buffer is not a Parameter, stays intact)
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    model = model.to(dev).freeze_backbone()
    opt = torch.optim.AdamW(model.obb_parameters(), lr=1e-3)

    scene = random_scene(6, F, lat_hw, D_APP, device=dev)
    obb = build_obb_tokens(scene, lat_hw, d_app=D_APP, cond_scale=4)      # varlen: no k_max
    plucker = build_plucker(scene.camera, (F,) + lat_hw)[None].to(dev)    # camera Plücker
    x1 = torch.randn(1, 48, F, H, W, device=dev)
    ctx = [torch.randn(8, 4096, device=dev)]
    print("tokens/frame (变长):", (obb.cu[1:] - obb.cu[:-1]).tolist())

    for i in range(5):
        x0 = torch.randn_like(x1)
        t01 = torch.rand(1, device=dev)
        xt = (1 - t01)[:, None, None, None, None] * x0 + t01[:, None, None, None, None] * x1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model([u for u in xt], t01 * 1000, ctx, S, obb=obb, plucker=plucker)
            L = flow_matching_loss(torch.stack(v).float(), x0.float(), x1.float())
        opt.zero_grad(); L.backward(); opt.step()
        print(f"step {i}  flow {L.item():.4f}")

    assert math.isfinite(L.item()), "loss not finite"
    go = model.obb_injectors[0].o.weight.grad
    assert go is not None and go.abs().sum() > 0, "OBB injector got no grad on real weights"
    g_emb = model.obb_embedder.id_proj.weight.grad
    assert g_emb is not None and g_emb.abs().sum() > 0, "OBB embedder got no grad (after o-proj moved)"
    assert model.patch_embedding.weight.grad is None, "backbone received grad (not frozen)"
    print("\nREAL Wan2.2-TI2V-5B + OBB OK: backbone frozen, OBB injector+embedder train, "
          f"velocity shape {tuple(v[0].shape)}")


if __name__ == "__main__":
    main()
