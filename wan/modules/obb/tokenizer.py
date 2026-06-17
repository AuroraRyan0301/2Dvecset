"""Build the per-pixel occlusion-stack OBB tokens, packed flat for variable-length attention.

Per covered pixel -> one token: fractional (t,h,w) position [for shared RoPE], ray->OBB entry depth,
the OBB id, and optional appearance. A pixel hit by N boxes -> N tokens (the occlusion stack); the
count varies per frame. Tokens are packed into ONE flat sequence with `cu` (cu_seqlens) marking the
per-frame boundaries -> flash-attn varlen (block-diagonal per frame). No k_max, no padding, no mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .obb import Scene
from .projection import project_obb_frame


@dataclass
class OBBTokens:
    """Flat (variable-length) OBB tokens for one scene. N = total tokens over all T frames."""
    pos: torch.Tensor          # (N, 3) float (t, h, w) fractional grid coords (shared RoPE)
    depth: torch.Tensor        # (N,)   float camera-z entry depth (scale -> Mellin)
    normal: torch.Tensor       # (N, 3) float world-frame face normal (unit; per-pixel surface orientation)
    instance_id: torch.Tensor  # (N,)   long  OBB index (table-free RFF id)
    appearance: torch.Tensor   # (N, D_app) float
    app_given: torch.Tensor    # (N,)   bool  True = appearance provided (else null_app)
    cu: torch.Tensor           # (T+1,) int32 cu_seqlens: frame f tokens = [cu[f] : cu[f+1]]

    @property
    def num_frames(self) -> int:
        return self.cu.numel() - 1

    def to(self, device):
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f).to(device))
        return self


def build_obb_tokens(scene: Scene, latent_hw, d_app: int, cond_scale: int = 8) -> OBBTokens:
    T = scene.num_frames
    dev = scene.camera.K.device
    frames = [[] for _ in range(T)]
    for oi, obb in enumerate(scene.objects):
        has_app = obb.appearance is not None
        app_vec = obb.appearance.to(dev) if has_app else torch.zeros(d_app, device=dev)
        for t in range(T):
            hit = project_obb_frame(obb, scene.camera, t, latent_hw, cond_scale)
            if hit is None:
                continue
            uvf, M = hit["uv_frac"], hit["uv_frac"].shape[0]
            frames[t].append({
                "pos": torch.stack([torch.full((M,), float(t), device=dev),
                                    uvf[:, 1], uvf[:, 0]], -1),         # (t, h=v, w=u) fractional
                "depth": hit["depth"],
                "normal": hit["normal"],                               # (M,3) world-frame face normal
                "instance_id": torch.full((M,), oi, device=dev, dtype=torch.long),
                "appearance": app_vec[None].expand(M, d_app),
                "app_given": torch.full((M,), has_app, device=dev, dtype=torch.bool),
            })

    def cat(key, width=None):
        parts = [f[key] for tf in frames for f in tf]
        if parts:
            return torch.cat(parts, 0)
        return torch.zeros((0, width) if width else (0,), device=dev)

    counts = torch.tensor([sum(f["depth"].shape[0] for f in tf) for tf in frames],
                          device=dev, dtype=torch.int32)
    cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
    return OBBTokens(
        pos=cat("pos", 3),
        depth=cat("depth"),
        normal=cat("normal", 3),
        instance_id=cat("instance_id").long(),
        appearance=cat("appearance", d_app),
        app_given=cat("app_given").bool(),
        cu=cu,
    )
