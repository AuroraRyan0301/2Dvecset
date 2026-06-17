"""Synthetic scenes + free GT maps (condition coarse, supervise rich).

`random_scene` builds a random OBB scene for mechanical tests; `render_gt_maps` reuses the
projection to get the front-most instance id + depth per pixel — the "free" supervision a
Blender/Infinigen renderer would give. (Real data: implement a dataset that yields the same Scene +
GT maps; the rest of the pipeline is identical.)
"""
from __future__ import annotations

import torch

from .obb import OBB, Camera, Scene
from .projection import project_obb_frame


def render_gt_maps(scene: Scene, latent_hw):
    """Front-most instance index + depth per latent pixel. Returns (gt_inst, gt_depth, fg)."""
    T = scene.num_frames
    H, W = latent_hw
    dev = scene.camera.K.device
    gt_inst = torch.zeros(T, H, W, dtype=torch.long, device=dev)
    gt_depth = torch.zeros(T, H, W, device=dev)
    best = torch.full((T, H, W), float("inf"), device=dev)
    fg = torch.zeros(T, H, W, dtype=torch.bool, device=dev)
    for oi, obb in enumerate(scene.objects):
        for t in range(T):
            hit = project_obb_frame(obb, scene.camera, t, latent_hw)
            if hit is None:
                continue
            u, v, d = hit["uv"][:, 0], hit["uv"][:, 1], hit["depth"]
            closer = d < best[t, v, u]
            v2, u2, d2 = v[closer], u[closer], d[closer]
            best[t, v2, u2] = d2
            gt_inst[t, v2, u2] = oi
            gt_depth[t, v2, u2] = d2
            fg[t, v2, u2] = True
    return gt_inst, gt_depth, fg


def random_scene(n_obj, T, latent_hw, d_app, device="cpu"):
    H, W = latent_hw
    K = torch.tensor([[150., 0., W * 8.], [0., 150., H * 8.], [0., 0., 1.]], device=device)
    cam = Camera(K=K,
                 R_cw=torch.eye(3, device=device)[None].expand(T, 3, 3).contiguous(),
                 t_cw=torch.tensor([0., 0., 5.], device=device)[None].expand(T, 3).contiguous(),
                 image_hw=(H * 16, W * 16))
    objs = []
    for _ in range(n_obj):
        c0 = (torch.rand(3, device=device) - 0.5) * torch.tensor([3., 2., 2.], device=device)
        vel = (torch.rand(3, device=device) - 0.5) * 0.3
        center = c0[None] + vel[None] * torch.arange(T, device=device)[:, None]
        rot6d = (torch.tensor([1., 0., 0., 0., 1., 0.], device=device)
                 + 0.1 * torch.randn(6, device=device))[None].expand(T, 6).contiguous()
        objs.append(OBB(center=center, rot6d=rot6d,
                        size=0.6 + 0.6 * torch.rand(3, device=device),
                        appearance=torch.randn(d_app, device=device)))
    return Scene(objects=objs, camera=cam, num_frames=T)
