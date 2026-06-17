# OBB-driven video generation on Wan2.2

This repo is a **Wan2.2 fork** with one added conditioning task — **`obb`** — that drives video
generation from a sparse set of **oriented 3D bounding boxes (OBBs)** projected by a known camera.
Everything follows Wan2.2's own structure (model subclass + config + task name), exactly like the
built-in `s2v` / `animate` tasks.

## Idea (one line)

OBBs are a **low-dimensional, LLM-authorable proxy** for a coarse G-buffer. The known camera projects
them into per-frame **per-pixel tokens** (occlusion-preserving: a pixel hit by N boxes → N
depth-tagged tokens, variable count per frame); these are injected into the frozen Wan DiT as a
**parallel cross-attention** (zero-init, like s2v's audio injector). The video model imagines all
detail inside the boxes.

> Why this and not ControlNet/CineMaster: those rasterize boxes to one dense depth map (occlusion
> collapsed, no orientation). We keep a sparse, occlusion-preserving, oriented token set. See
> `docs/DESIGN.md` and `docs/RELATED_WORK.md` (code-verified competitor map).

## Layout (Wan2.2 + the OBB task)

```
wan/                              Wan2.2 source (backbone, VAE, T5, pipelines)
  modules/obb/                    ← the OBB task
    model_obb.py                  WanModel_OBB(WanModel): re-impl forward + inject OBB cross-attn
    injector.py                   OBBTokenEmbedder + OBBCrossAttention (zero-init, shared RoPE)
    obb.py projection.py          OBB/Camera/Scene + project OBB -> per-pixel footprint + depth
    tokenizer.py                  build the variable-length occlusion-stack tokens
    rope.py                       Wan-compatible 3D RoPE at arbitrary (t,h,w)
    heads.py losses.py            self-tracking head; rectified-flow + self-tracking losses
    data.py                       synthetic scenes + free GT instance/depth maps
  configs/wan_obb_5B.py           task config, registered as 'obb-5B' in WAN_CONFIGS
docs/                             DESIGN / ARCHITECTURE / RELATED_WORK
scripts/                         test_obb_model.py (tiny GPU), test_obb_5b.py (real TI2V-5B)
_refsrc/                          reference clones + downloaded ckpts (Wan2.2-TI2V-5B, ...)
```

## Status

**Working & verified on real weights** (`scripts/test_obb_5b.py`, H100): Wan2.2-TI2V-5B loads
(backbone weights all matched; OBB params new), backbone frozen, OBB injector+embedder train,
velocity output shape correct, variable tokens/frame. See `docs/ARCHITECTURE.md`.

**TODO (still Wan-native):** `wan/obb2video.py` pipeline + `generate.py` dispatch; real data
(Wan-VAE encode video → target latent, T5 → text); expose features from `model_obb` to wire the
self-tracking loss; training script; port to A14B (inject into both MoE experts);
tokenizer pad → `flash_attn_varlen` for unbounded per-frame length.

## Run the tests

```bash
PY=/gs/fs/tga-koike-shanda4/yurh/miniconda3/envs/trellis2/bin/python   # torch 2.6 + diffusers etc.
# real Wan2.2-TI2V-5B integration (loads ckpt, bf16):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD "$PY" scripts/test_obb_5b.py
# tiny random-weight wiring check:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD "$PY" scripts/test_obb_model.py
```
Env note: the trellis2 env was extended with `diffusers ftfy decord dashscope librosa soundfile`.
`wan/__init__.py` imports task pipelines optionally, so missing s2v/animate deps don't block `obb`.
