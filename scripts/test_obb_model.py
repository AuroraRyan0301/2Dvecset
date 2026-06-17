"""Verify WanModel_OBB (Wan2.2 subclass) wiring on GPU with a TINY random-weight model.

Checks the Wan-native integration: forward runs with OBB injection, output shape matches the latent,
backbone frozen, OBB branch gets gradient. (Tiny config; the real run loads TI2V-5B weights.)

Run:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD <trellis2-python> scripts/test_obb_model.py
"""
import torch

from wan.configs import WAN_CONFIGS                      # checks task registration ('obb-5B')
from wan.modules.obb import WanModel_OBB, random_scene, build_obb_tokens, build_plucker

print("registered:", "obb-5B" in WAN_CONFIGS)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    F, H, W = 4, 16, 16                                   # latent video; token grid (4,8,8), S=256
    lat_hw, S = (H // 2, W // 2), F * (H // 2) * (W // 2)

    model = WanModel_OBB(
        model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=128, ffn_dim=256,
        freq_dim=256, text_dim=128, out_dim=16, num_heads=4, num_layers=4,
        obb_inject_layers=[0, 2],
        obb_d_app=8,
    ).to(dev).freeze_backbone()
    # WanModel zero-inits its output head -> a RANDOM model outputs 0 -> all grads 0. Un-zero it so
    # this wiring test is meaningful. (Real TI2V-5B weights have a trained, nonzero head.)
    torch.nn.init.normal_(model.head.head.weight, std=0.02)

    assert any(p.requires_grad for p in model.obb_parameters()), "OBB not trainable"
    backbone_ids = {id(p) for p in model.obb_parameters()}
    assert all(not p.requires_grad for p in model.parameters() if id(p) not in backbone_ids), "backbone not frozen"

    scene = random_scene(5, F, lat_hw, d_app=8, device=dev)
    obb = build_obb_tokens(scene, lat_hw, d_app=8, cond_scale=4)           # varlen: no k_max
    plucker = build_plucker(scene.camera, (F,) + lat_hw)[None].to(dev)     # camera Plücker
    x = [torch.randn(16, F, H, W, device=dev)]
    ctx = [torch.randn(8, 128, device=dev)]

    with torch.autocast("cuda", dtype=torch.bfloat16):    # flash-attn needs half precision
        v = model(x, torch.rand(1, device=dev) * 1000, ctx, S, obb=obb, plucker=plucker)
    print("velocity out:", tuple(v[0].shape), "| tokens/frame:", (obb.cu[1:] - obb.cu[:-1]).tolist())
    assert v[0].shape == (16, F, H, W), v[0].shape

    v[0].float().abs().sum().backward()
    # zero-init output proj => on the FIRST backward only the injector's o-proj gets grad
    # (embedder is upstream of the zero proj; it starts learning once o-proj moves -- see MiniWan test).
    go = model.obb_injectors[0].o.weight.grad
    assert go is not None and go.abs().sum() > 0, "OBB injector got no grad (branch not in graph)"
    assert model.patch_embedding.weight.grad is None, "backbone received grad (not frozen)"
    print("WanModel_OBB OK: forward+OBB inject runs, output shape matches, backbone frozen, OBB branch trains")


if __name__ == "__main__":
    main()
