"""Wan-compatible 3D RoPE at arbitrary (possibly fractional) per-token (t, h, w) positions.

Wan splits head_dim=128 (64 complex freqs) across (t, h, w) = (22, 21, 21) complex (all used; no
spare axis for depth -- see docs/ARCHITECTURE.md). We reuse the SAME split so OBB condition tokens
live in the same rotary frame as the video latent patches, and bind to them geometrically.

Positions are applied CONTINUOUSLY (angle = pos * freq, computed on the fly) rather than by integer
table lookup, so an OBB token may sit at a fractional grid coordinate (pixel / stride, not rounded)
-- same coordinate convention as the video tokens, so the two share one RoPE frame at any resolution
(resolution changes are then handled exactly like the backbone: position interpolation on the shared
grid). Depth/r are carried as features, NOT RoPE axes here (video Q has no depth; see injector.py).
"""
from __future__ import annotations

import torch


def build_3d_freqs(head_dim: int = 128, theta: float = 10000.0) -> tuple:
    """Wan's split: c = head_dim/2 freqs partitioned [c-2*(c//3), c//3, c//3] for (t,h,w).
    Returns a 3-tuple of 1-D frequency vectors (lengths sum to c); freq_i = 1/theta^(i/s)."""
    c = head_dim // 2
    splits = [c - 2 * (c // 3), c // 3, c // 3]            # (22, 21, 21) for 128
    out = []
    for s in splits:
        idx = torch.arange(s, dtype=torch.float32)
        out.append(1.0 / (theta ** (idx / s)))            # (s,)
    return tuple(out)


def apply_rope_positions(x: torch.Tensor, pos: torch.Tensor, freqs_thw: tuple) -> torch.Tensor:
    """Apply 3D RoPE to `x` at explicit (fractional-ok) positions.

    x:   (B, S, n_heads, head_dim)  real
    pos: (B, S, 3)                  float (t, h, w) positions (integer or fractional)
    freqs_thw: 3-tuple of 1-D freq vectors from build_3d_freqs (axis lengths sum to head_dim/2).
    """
    B, S, H, D = x.shape
    xc = torch.view_as_complex(x.float().reshape(B, S, H, D // 2, 2))      # (B,S,H,D/2)
    ang = torch.cat([pos[..., a:a + 1].float() * freqs_thw[a].to(pos.device)
                     for a in range(3)], dim=-1)                          # (B,S,D/2)
    rot = torch.polar(torch.ones_like(ang), ang).unsqueeze(2)             # (B,S,1,D/2) complex
    out = torch.view_as_real(xc * rot).reshape(B, S, H, D)
    return out.type_as(x)
