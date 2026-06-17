"""Project an OBB through the known camera to its covered pixels + entry depth.

The 'free coarse G-buffer' step (docs/DESIGN.md §1, §3): no authored per-pixel data — projecting the
low-dim box yields, for every covered pixel, the ray->OBB entry depth. Overlapping boxes give
overlapping covered sets at different depths -> the occlusion *stack*.

Rasterized at a CONDITION resolution = cond_scale × the latent grid (finer than the latent), so each
covered pixel has a genuinely FRACTIONAL latent coordinate `(pixel+0.5)/cond_scale` — the sub-cell
position moves continuously with sub-pixel box motion (no box-center hack needed). This mirrors how an
image condition (DINO/VAE) is finer than the latent and thus sub-pixel sensitive. cond_scale trades
fidelity vs token count (cond_scale=8 -> ~1/8-cell granularity).
"""
from __future__ import annotations

import torch

from .obb import OBB, Camera


def project_obb_frame(obb: OBB, cam: Camera, frame: int, latent_hw, cond_scale: int = 8,
                      eps: float = 1e-6):
    """Return covered pixels (at cond resolution) + entry depth for one OBB in one frame.

    Returns dict with:
      'uv'      (M, 2) long  latent CELL coords [u, v]            (for GT rasterization)
      'uv_frac' (M, 2) float fractional latent coords [u, v] = (cond_pixel + 0.5) / cond_scale
      'depth'   (M,)   float camera-z entry depth
      'normal'  (M, 3) float WORLD-frame normal of the OBB face the ray enters (unit; per pixel, so
                       different faces of one box carry different normals -> per-pixel surface
                       orientation, a G-buffer normal channel that depth alone lacks).
    None if the box does not project into the frame.
    """
    Hl, Wl = latent_hw
    Hc, Wc = Hl * cond_scale, Wl * cond_scale         # condition grid (finer than latent)
    _, W_img = cam.image_hw
    stride = W_img / Wc                               # image px per cond pixel

    K = cam.K
    R_cw, t_cw = cam.R_cw[frame], cam.t_cw[frame]
    center, R_box, half = obb.center[frame], obb.matrices()[frame], obb.half

    center_cam = R_cw @ center + t_cw                # OBB center in cam frame
    axes_cam = R_cw @ R_box                          # box axes in cam frame (columns)

    # 8 corners -> pixels -> candidate AABB in the cond grid
    signs = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                         dtype=center.dtype, device=center.device)
    corners_cam = center_cam[None] + (signs * half) @ axes_cam.T      # (8,3)
    if (corners_cam[:, 2] <= 0).all():
        return None                                                  # fully behind camera
    z = (K @ corners_cam.T).T[:, 2:3].clamp_min(eps)
    px = (K @ corners_cam.T).T[:, :2] / z                            # (8,2) image px
    uvc = px / stride                                                # cond-grid coords
    u0 = int(torch.floor(uvc[:, 0].min()).clamp(0, Wc - 1))
    u1 = int(torch.ceil(uvc[:, 0].max()).clamp(0, Wc - 1))
    v0 = int(torch.floor(uvc[:, 1].min()).clamp(0, Hc - 1))
    v1 = int(torch.ceil(uvc[:, 1].max()).clamp(0, Hc - 1))
    if u1 < u0 or v1 < v0:
        return None

    # rays through cond-pixel centers (cam frame)
    us = torch.arange(u0, u1 + 1, device=center.device)
    vs = torch.arange(v0, v1 + 1, device=center.device)
    gv, gu = torch.meshgrid(vs, us, indexing="ij")
    su, sv = gu.flatten(), gv.flatten()
    pix = torch.stack([(su.float() + 0.5) * stride, (sv.float() + 0.5) * stride,
                       torch.ones_like(su, dtype=center.dtype)], -1)   # (N,3)
    d_cam = pix @ torch.linalg.inv(K).T                                # (N,3) ray dirs

    # ray -> box-local slab test (cam origin = 0)
    o_local = -(axes_cam.T @ center_cam)
    d_local = d_cam @ axes_cam
    inv_d = 1.0 / torch.where(d_local.abs() < eps, torch.full_like(d_local, eps), d_local)
    t1 = (-half[None] - o_local[None]) * inv_d
    t2 = (half[None] - o_local[None]) * inv_d
    tmin = torch.minimum(t1, t2)
    t_near = tmin.max(dim=-1).values
    t_far = torch.maximum(t1, t2).min(dim=-1).values
    k_star = tmin.argmax(dim=-1)                                       # (N,) entry-face box axis
    depth = (t_near.clamp_min(0)[:, None] * d_cam)[:, 2]               # camera-z at entry
    hit = (t_near <= t_far) & (t_far > 0) & (depth > 0)
    if hit.sum() == 0:
        return None

    su, sv, depth = su[hit], sv[hit], depth[hit]
    ks, dl = k_star[hit], d_local[hit]                                 # entry axis + ray dir (local)
    sgn = -torch.sign(dl.gather(1, ks[:, None]).squeeze(1))            # outward face faces the ray
    normal = sgn[:, None] * R_box[:, ks].T                            # (M,3) world-frame face normal (unit)
    uv_frac = torch.stack([(su.float() + 0.5) / cond_scale,
                           (sv.float() + 0.5) / cond_scale], -1)       # (M,2) fractional latent coords
    uv = torch.stack([su // cond_scale, sv // cond_scale], -1).long()  # (M,2) latent cell (for GT)
    return {"uv": uv, "uv_frac": uv_frac, "depth": depth, "normal": normal}
