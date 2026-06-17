"""OBB per-pixel token embedder + the parallel cross-attention injector (used by WanModel_OBB).

Per-pixel token = table-free OBB id (random Fourier) + Mellin(entry depth) + optional appearance
(null_app if not given). The cross-attention mirrors Wan's parallel image cross-attn (zero-init
output): Q = video tokens (integer (t,h,w) positions), K/V = the OBB tokens (FRACTIONAL (t,h,w)
positions = pixel/stride). Both get the SAME shared (t,h,w) RoPE, so an OBB token binds to the video
cell it projects into; the fractional K positions make the attention phase shift continuously under
sub-pixel box motion. Attention is frame-local + variable-length via flash-attn varlen (cu_seqlens):
video Q is uniform HW per frame, OBB K/V is packed per-frame by `cu`. Zero compute on absent frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func

from .rope import apply_rope_positions
from .tokenizer import OBBTokens


def mellin(x, bands, eps=1e-3):
    """Fourier(log|x|): characters of the scale group (global scale -> additive shift)."""
    lx = torch.log(x.abs().clamp_min(eps))
    freqs = (2.0 ** torch.arange(bands, device=x.device, dtype=lx.dtype))
    a = lx[..., None] * freqs
    return torch.cat([a.sin(), a.cos()], dim=-1)


class OBBTokenEmbedder(nn.Module):
    """Flat OBB tokens -> (N, dim): id + Mellin(depth) + world-frame face normal + optional appearance."""

    def __init__(self, dim, d_app, id_freqs=64, id_scale=8.0, depth_bands=8):
        super().__init__()
        self.depth_bands = depth_bands
        self.depth_proj = nn.Linear(2 * depth_bands, dim)            # depth scale -> Mellin
        self.normal_proj = nn.Linear(3, dim)                         # world-frame face normal (raw unit 3-vec)
        self.register_buffer("id_freqs", torch.randn(id_freqs) * id_scale, persistent=True)
        self.id_proj = nn.Linear(2 * id_freqs, dim)                  # table-free OBB id (random Fourier)
        self.app_proj = nn.Linear(d_app, dim)
        self.null_app = nn.Parameter(torch.zeros(dim))               # "no appearance given"
        self.norm = nn.LayerNorm(dim)

    def _id_feat(self, idx, dt):
        a = idx.to(dt)[..., None] * self.id_freqs.to(dt)
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def forward(self, tok: OBBTokens) -> torch.Tensor:
        dt = self.depth_proj.weight.dtype
        app = torch.where(tok.app_given[..., None],
                          self.app_proj(tok.appearance.to(dt)), self.null_app.to(dt))
        e = (self.id_proj(self._id_feat(tok.instance_id, dt))
             + self.depth_proj(mellin(tok.depth.to(dt), self.depth_bands))
             + self.normal_proj(tok.normal.to(dt))
             + app)
        return self.norm(e)                                          # (N, dim)


class OBBCrossAttention(nn.Module):
    """One layer's OBB cross-attention; its (zero-init) output is added to the video stream.
    `freqs` (3D RoPE banks) is assigned by WanModel_OBB and shared across layers."""

    def __init__(self, dim, n_heads, qk_norm=True):
        super().__init__()
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)   # zero-init -> no-op at start
        self.norm_q = nn.LayerNorm(self.hd) if qk_norm else nn.Identity()
        self.norm_k = nn.LayerNorm(self.hd) if qk_norm else nn.Identity()
        self.freqs = None                                            # set by WanModel_OBB

    def _video_positions(self, F_, H_, W_, device):
        t = torch.arange(F_, device=device).view(F_, 1, 1).expand(F_, H_, W_)
        h = torch.arange(H_, device=device).view(1, H_, 1).expand(F_, H_, W_)
        w = torch.arange(W_, device=device).view(1, 1, W_).expand(F_, H_, W_)
        return torch.stack([t, h, w], -1).reshape(F_ * H_ * W_, 3).float()

    def forward(self, x, fhw, obb_emb, obb_pos, cu_k):
        # x:(1,S,D) video; obb_emb:(N,D); obb_pos:(N,3); cu_k:(F+1,) int32 per-frame OBB boundaries
        S, D = x.shape[1], x.shape[2]
        F_, H_, W_ = fhw
        HW = H_ * W_
        nh, hd = self.n_heads, self.hd
        freqs = [f.to(x.device) for f in self.freqs]

        q = self.norm_q(self.q(x[0]).view(1, S, nh, hd))                          # video Q (integer pos)
        qpos = self._video_positions(F_, H_, W_, x.device).unsqueeze(0)
        q = apply_rope_positions(q, qpos, freqs)[0]                               # (S, nh, hd)

        N = obb_emb.shape[0]
        k = self.norm_k(self.k(obb_emb).view(1, N, nh, hd))                       # OBB K (fractional pos)
        k = apply_rope_positions(k, obb_pos.unsqueeze(0), freqs)[0]               # (N, nh, hd)
        v = self.v(obb_emb).view(N, nh, hd)

        cu_q = torch.arange(0, (F_ + 1) * HW, HW, device=x.device, dtype=torch.int32)
        max_k = int((cu_k[1:] - cu_k[:-1]).max()) if cu_k.numel() > 1 else 0
        h = torch.bfloat16                                                        # flash_attn needs fp16/bf16
        o = flash_attn_varlen_func(q.to(h).contiguous(), k.to(h).contiguous(), v.to(h).contiguous(),
                                   cu_q, cu_k.to(torch.int32), HW, max_k, causal=False)
        o = torch.nan_to_num(o.to(x.dtype))                                       # frames w/ 0 OBBs -> 0
        return self.o(o.reshape(1, S, D))
