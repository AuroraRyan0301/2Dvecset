"""OBB scene data structures and rotation utilities.

An OBB is the low-dimensional, LLM-authorable proxy (see docs/DESIGN.md §2): we never store the
object's true shape — only a rigid spacetime envelope (center + size + 6D rotation per frame) plus
an optional appearance latent. Objects are identified positionally by their index in Scene.objects
(that index becomes the table-free RFF id), so no explicit id/class field is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 2019 continuous 6D rotation rep -> (..., 3, 3). Columns are the basis vectors,
    so world = R @ local + center."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3), columns = axes


@dataclass
class OBB:
    """A single oriented box, possibly with a per-frame trajectory.

    Tensors are sized over T frames so static and dynamic objects share one code path
    (static = constant trajectory). All in world coordinates.
    """
    center: torch.Tensor        # (T, 3)
    rot6d: torch.Tensor         # (T, 6)  -> rotation_6d_to_matrix
    size: torch.Tensor          # (3,)    full extent (we use half = size/2)
    appearance: Optional[torch.Tensor] = None   # (D_app,) identity/look latent; None = not given

    @property
    def half(self) -> torch.Tensor:
        return self.size * 0.5

    def matrices(self) -> torch.Tensor:
        return rotation_6d_to_matrix(self.rot6d)  # (T, 3, 3)


@dataclass
class Camera:
    """Known camera (the projection is free, see docs/DESIGN.md §1)."""
    K: torch.Tensor             # (3, 3) intrinsics at full image resolution
    R_cw: torch.Tensor          # (T, 3, 3) world->cam rotation
    t_cw: torch.Tensor          # (T, 3)    world->cam translation
    image_hw: tuple             # (H_img, W_img) full resolution

    @property
    def num_frames(self) -> int:
        return self.R_cw.shape[0]


@dataclass
class Scene:
    """A whole video scene: a set of OBBs + a known camera."""
    objects: list                       # list[OBB]
    camera: Camera
    num_frames: int
