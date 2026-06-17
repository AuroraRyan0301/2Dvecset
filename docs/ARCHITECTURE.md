# Architecture — OBB task on the Wan2.2 backbone

Implemented as `wan/modules/obb/model_obb.py::WanModel_OBB(WanModel)`, registered as task `obb-5B`.
Mirrors how Wan2.2 adds `s2v`/`animate` (model subclass + config + task name).

## Wan2.2 TI2V-5B backbone (the verified target)

| | value |
|---|---|
| dim | 3072 |
| layers | 30 |
| heads | 24 (head_dim = 128) |
| ffn_dim | 14336 |
| in/out latent ch | 48 (Wan2.2 VAE) |
| patch_size | (1, 2, 2) → token grid (F, H/2, W/2) |
| vae_stride | (4, 16, 16) |
| text | umT5-xxl (text_dim 4096), cross-attention per block |
| objective | rectified flow |

Block: `WanAttentionBlock.forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens)` —
self-attn (3D RoPE) + text cross-attn + FFN + adaLN. (Same block as A14B's experts, so this ports
to A14B by injecting into both experts.)

## Injection (WanModel_OBB)

`forward` is WanModel's forward with one addition: at each `obb_inject_layers` layer, after the
frozen block, add a gated OBB cross-attention residual (see `wan/modules/obb/README.md` for the
math). Properties:
- **Q** = video tokens; **K/V** = OBB per-pixel stack tokens (variable length per frame).
- **Frame-local**: fold the temporal dim into batch → each frame's video tokens attend only that
  frame's OBB tokens → no `100×`-sequence blow-up; cost ∝ Σ_t (tokens_per_frame · K_t).
- **Shared (t,h,w) RoPE** on Q and K → geometric spatial binding (not learned-from-scratch).
- **Zero-init output proj** → no-op at init; backbone unchanged until the branch learns.
- Only `obb_parameters()` (embedder + injectors + gate) train; `freeze_backbone()` freezes the rest.

## RoPE — the catch

Wan's 3D RoPE fills head_dim=128 (64 complex freqs) across **(t,h,w)=(22,21,21) complex — all used,
no spare axis** (unlike FLUX). So **depth is a token feature** (Fourier(depth) in the embedder), not
a 4th RoPE axis. (A 4th axis would need re-partitioning head_dim + finetuning.)

## Practical notes (verified on H100)

- **Zero-init head**: a random `WanModel` zero-inits its output head → outputs ≡0 → all grads 0.
  Only real/trained weights show gradient. (`scripts/test_obb_model.py` un-zeros the head for the
  random-weight check; `scripts/test_obb_5b.py` uses real TI2V-5B weights.)
- **bf16 load** (~10GB, fits 34GB): cast only params to bf16; the complex RoPE buffer is not a
  Parameter and must stay intact. Run forward under `autocast('cuda', bf16)` (flash-attn needs half).
- **Loading**: `WanModel_OBB.load_state_dict(ckpt_sd, strict=False)` — backbone matches, OBB params
  are reported "missing" (newly initialised), as intended.
- **Env**: trellis2 python + `diffusers ftfy decord dashscope librosa soundfile`. `wan/__init__`
  imports task pipelines optionally so missing s2v/animate deps don't block `obb`.

## Long video & chunking

Per-frame OBB conditioning is orthogonal to chunk/autoregressive generation (CausVid/Self-Forcing/
Rolling-Forcing are Wan-based): chunking sets temporal causality; the OBB cross-attn still feeds each
frame its own boxes. Cross-chunk appearance consistency relies on the per-object appearance latent
(shared across frames/chunks) + backbone temporal attention.

## Training plan

Freeze backbone; train OBB params with rectified-flow loss + **self-tracking auxiliary loss**
(predict per-frame instance/depth from features — needs `model_obb` to expose the last hidden, a
TODO). Condition on loose boxes, supervise with the free synthetic GT masks ("condition coarse,
supervise rich"). Curriculum: ~50 boxes, no appearance/BVH → +appearance latent → +BVH → chunked
long video. Headline benchmark: dense-occlusion video vs per-frame SeeThrough3D + CineMaster.
