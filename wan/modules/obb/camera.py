"""Camera conditioning: per-pixel Plücker rays -> encoder -> added to video tokens.

The mainstream recipe (CameraCtrl / Wan camera control): represent each ray as a 6-D Plücker
coordinate (d, o×d) -- direction + moment -- which pins the 3D ray (rotation in d, translation in
the moment), is numerically uniform, and is geometrically meaningful per pixel. Encoded and added
to the video tokens before the blocks (zero-init -> no-op at start).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_plucker(camera, fhw):
    """Per-token-cell Plücker (d, o×d) in world frame at the token grid. Returns (S, 6), S=F·Ht·Wt."""
    F_, Ht, Wt = fhw
    dev = camera.K.device
    _, W_img = camera.image_hw
    stride = W_img / Wt
    Kinv = torch.linalg.inv(camera.K)
    vs = torch.arange(Ht, device=dev)
    us = torch.arange(Wt, device=dev)
    gv, gu = torch.meshgrid(vs, us, indexing="ij")
    px = (gu.flatten().float() + 0.5) * stride
    py = (gv.flatten().float() + 0.5) * stride
    d_cam = torch.stack([px, py, torch.ones_like(px)], -1) @ Kinv.T     # (HW,3) cam-frame ray dirs
    out = []
    for f in range(F_):
        R, t = camera.R_cw[f], camera.t_cw[f]                          # world->cam
        d_world = d_cam @ R                                            # cam->world rotate (= R^T d)
        d_world = d_world / d_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        o_world = -(t @ R)                                             # camera center in world (3,)
        m = torch.cross(o_world[None].expand_as(d_world), d_world, dim=-1)
        out.append(torch.cat([d_world, m], -1))                        # (HW,6)
    return torch.stack(out, 0).reshape(F_ * Ht * Wt, 6)                # (S,6)


class CameraEncoder(nn.Module):
    """Plücker (B,S,6) -> (B,S,dim); zero-init output -> no-op at init."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, plucker):
        return self.net(plucker)
