# `wan/modules/obb` — the OBB conditioning task

`WanModel_OBB(WanModel)` re-implements `forward` and injects OBB conditioning as a parallel
cross-attention at selected layers (Wan-native, like s2v). Backbone frozen; only `obb_parameters()`
train.

## Files

| file | what |
|---|---|
| `model_obb.py` | `WanModel_OBB`: embed → per-layer cross-attn inject; `obb_parameters()`, `freeze_backbone()` |
| `injector.py`  | `OBBTokenEmbedder` + `OBBCrossAttention` (zero-init, shared (t,h,w) RoPE, flash varlen) + `mellin` |
| `camera.py`    | `build_plucker` (per-pixel Plücker) + `CameraEncoder` (→ added to video tokens) |
| `obb.py` `projection.py` | OBB/Camera/Scene; ray–OBB → covered pixels + entry depth (rasterized at cond res) |
| `tokenizer.py` | flat occlusion-stack tokens + `cu` (cu_seqlens): pos(t,h,w), depth, id, appearance |
| `rope.py` `heads.py` `losses.py` `data.py` | continuous 3D RoPE; self-tracking head; losses; synthetic data + GT |

## Per-pixel token (minimal, by design)

```
e = id_proj(RandFourier(id)) + depth_proj(Mellin(depth)) + (appearance | null_app)   → LayerNorm
```
Why so small: in the camera frame the entry point = `depth · ray_dir`, and `ray_dir` is fixed by the
pixel = the **(t,h,w) RoPE position**; the only non-redundant per-pixel geometry is **depth** (scale →
**Mellin** = Fourier(log), no normalization). Direction lives in RoPE. World-frame coords dropped
(recoverable by the backbone from depth + camera Plücker; re-add if motion-disentangling needs it).

## Flow

```
OBB ─project(cond_scale)→ covered pixels {pos(t,h,w fractional), depth}   (flat tokens + cu_seqlens)
        │ ★OBBTokenEmbedder   e = id(RFF) + Mellin(depth) + appearance|null
        ▼  obb_emb (N, dim)
        │ ★OBBCrossAttention (flash varlen)  Q=video(integer pos), K/V=obb(fractional pos),
        │   shared (t,h,w) RoPE, block-diagonal per frame (cu_q / cu_k), zero-init out
        ▼  added (gated) to the frozen Wan stream at obb_inject_layers (default every-other block)

camera: K,RT ─build_plucker→ rays(6-D) ─★CameraEncoder→ added to video tokens (once, at input)
```
No camera interaction inside the condition (the projection already used it; the video-side Plücker
gives the backbone the viewpoint). No scene encoder — OBB tokens go straight to cross-attn (deleted
2026-06-17; the occlusion stack is now resolved by the backbone alone, see caveats).

## Position: integer video Q, fractional OBB K (shared RoPE)

Video Q sits at **integer** grid coords; OBB K sits at **fractional** coords `= (cond_pixel+0.5)/cond_scale`.
The footprint is rasterized at `cond_scale × latent` resolution (default 8), so each covered pixel gets
a genuine fractional latent coordinate — same mechanism as an image condition (DINO/VAE) being finer
than the latent. Both share the **same continuous (t,h,w) RoPE** (`rope.py` applies `angle = pos·freq`
on the fly, fractional-ok), so an OBB token binds to the video cell it projects into, and the Q·K phase
`(pos_Q − pos_K)·ω` shifts *continuously* with sub-pixel box motion (integer-only K would let many
phases cancel exactly). Resolution-agnostic for free: OBB shares the video grid, so it inherits the
backbone's resolution handling (position interpolation / NTK) — no `x/res` normalization (that's the
GLIGEN/DETR box-coord-as-content recipe, a different, learned-binding mechanism).

## Conditional / unconditional (CFG)

Per-OBB **optional appearance** (`app_given` → `null_app`). Train with condition-dropout to null
(per-OBB → `null_app`; whole condition → `forward(obb=None)`); infer with guidance
`ε=ε(∅)+s·(ε(c)−ε(∅))`. Null is a *learned* embedding.

## Notes / honest caveats

- **TRUE varlen, no k_max**: tokens are packed flat `(N, …)` with `cu` (per-frame cu_seqlens);
  `flash_attn_varlen_func` runs block-diagonal per frame. No padding, no mask, no per-frame cap.
  Cost ∝ Σ (HWᵢ · nᵢ) over real per-frame OBB counts nᵢ. `flash_attn_varlen_func` needs fp16/bf16 —
  q/k/v are cast to bf16 before the call (Wan's wrapper auto-casts; the raw func does not).
- **cond_scale is a real cost knob, not free fidelity**: it multiplies token count by ~cond_scale².
  8× → up to 64× the latent footprint in tokens. Must be swept empirically; default 8 is a guess.
- **Fractional position buys smoothness, maybe not control**: the 64 cond-pixels inside one latent
  cell share near-identical id/depth content and differ only by ≤1 cell in position → highly
  redundant. Fractional pos mainly makes the gradient w.r.t. box pose smooth/differentiable; do NOT
  oversell it as "sub-pixel control" (the latent renders at ÷32 regardless).
- **Occlusion is now backbone-only**: with the scene encoder gone, "same pixel, different depth"
  tokens no longer interact explicitly — the backbone must separate them from the K set. If occluded
  regions look muddy, re-adding a small same-cell depth interaction is the first thing to try.
- **GT cell vs token cell mismatch (latent bug to check when wiring self-tracking)**: token position
  uses `(su+0.5)/cond_scale`, GT (`render_gt_maps`) uses `su // cond_scale` for the latent cell —
  `round` vs `floor` can disagree by one cell at boundaries. Not exercised until GT loss is wired.
- **Zero-init head**: a *random* `WanModel` outputs 0 → all grads 0; use real weights
  (`scripts/test_obb_5b.py`).
- **bf16 load**: cast only params (complex RoPE buffer is not a Parameter). Forward under
  `autocast('cuda', bf16)`. **Token grid**: `latent_hw` = post-patchify `(H/2,W/2)`.
```
