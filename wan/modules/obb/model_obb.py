"""WanModel_OBB: Wan2.2 backbone + OBB conditioning injected as parallel cross-attention.

Follows the Wan2.2 conditioning-task pattern (cf. s2v's AudioInjector / animate): subclass WanModel,
add a token embedder + per-layer cross-attention injectors (zero-init output -> no-op at init),
re-implement forward and add a gated residual at selected layers.

Each OBB pixel-token (variable length per frame, occlusion-preserving) is K/V; video tokens are Q;
shared (t,h,w) RoPE binds them; depth is a token feature. Backbone frozen, only `obb_parameters()`
train.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..model import WanModel, sinusoidal_embedding_1d
from .injector import OBBTokenEmbedder, OBBCrossAttention
from .camera import CameraEncoder
from .rope import build_3d_freqs
from .tokenizer import OBBTokens


class WanModel_OBB(WanModel):
    def __init__(self, *args,
                 obb_inject_layers=None, obb_d_app=32, obb_id_freqs=64, obb_depth_bands=8, **kwargs):
        super().__init__(*args, **kwargs)
        L = self.num_layers
        self.obb_inject_layers = list(range(0, L, 2)) if obb_inject_layers is None else list(obb_inject_layers)
        self.obb_layers_map = {l: i for i, l in enumerate(self.obb_inject_layers)}
        self.obb_embedder = OBBTokenEmbedder(self.dim, obb_d_app, id_freqs=obb_id_freqs,
                                             depth_bands=obb_depth_bands)
        self.cam_encoder = CameraEncoder(self.dim)                  # Plücker camera -> video tokens
        self.obb_injectors = nn.ModuleList(
            [OBBCrossAttention(self.dim, self.num_heads, qk_norm=True) for _ in self.obb_inject_layers])
        freqs = build_3d_freqs(self.dim // self.num_heads)         # shared (t,h,w) RoPE banks
        for inj in self.obb_injectors:
            inj.freqs = freqs
        self.obb_gate = nn.Parameter(torch.ones(len(self.obb_inject_layers)))
        # OBBCrossAttention.o is zero-init -> the whole branch is a no-op at init.

    # ---- training helpers ----
    def obb_modules(self):
        return [self.obb_embedder, self.obb_injectors, self.cam_encoder]

    def obb_parameters(self):
        ps = [self.obb_gate]
        for m in self.obb_modules():
            ps += list(m.parameters())
        return ps

    def freeze_backbone(self):
        keep = {id(p) for p in self.obb_parameters()}
        for p in self.parameters():
            p.requires_grad_(id(p) in keep)
        return self

    # ---- forward = WanModel.forward + OBB injection at selected layers ----
    def forward(self, x, t, context, seq_len, obb: OBBTokens = None, plucker=None, y=None):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        if plucker is not None:                                    # camera: Plücker -> add to video tokens
            cam = self.cam_encoder(plucker)
            if cam.size(1) < seq_len:
                cam = torch.cat([cam, cam.new_zeros(cam.size(0), seq_len - cam.size(1), cam.size(2))], dim=1)
            x = x + cam

        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        context = self.text_embedding(torch.stack(
            [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                      freqs=self.freqs, context=context, context_lens=None)

        # OBB conditioning: embed flat per-pixel tokens -> per-layer varlen cross-attn (single scene)
        obb_emb = None
        if obb is not None:
            obb_emb = self.obb_embedder(obb)                       # (N, dim)
        fhw = tuple(int(s) for s in grid_sizes[0].tolist())        # assumes uniform grid in batch
        P = fhw[0] * fhw[1] * fhw[2]

        for i, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            if obb_emb is not None and i in self.obb_layers_map:
                j = self.obb_layers_map[i]
                res = self.obb_injectors[j](x[:, :P], fhw, obb_emb, obb.pos, obb.cu)     # (1,P,dim)
                if P < seq_len:
                    res = torch.cat([res, res.new_zeros(res.size(0), seq_len - P, res.size(2))], dim=1)
                x = x + self.obb_gate[j] * res

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
