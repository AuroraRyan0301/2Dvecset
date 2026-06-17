# Related Work — code‑verified competitor map

All findings below were verified at the **code level** (repos cloned to `_refsrc/`) or, where no
code exists, from the paper. Verified 2026‑06‑16.

## Representation taxonomy

Conditioning representations split into three families; our method is in none of them:

- **Dense rendered 2D image → conv/VAE → (residual‑add | adaLN | channel‑concat).** Fixed‑length
  (resolution‑determined), occlusion collapsed to one value/pixel.
- **Per‑object coordinate token → (cross | gated‑self) attention.** Variable over object count but
  **padded to a fixed small cap**; one token per object regardless of footprint; no occlusion.
- **Per‑pixel occlusion‑stack token (OURS).** Token budget = coverage × overlap depth; multiple
  depth‑tagged tokens per pixel; content‑adaptive, variable per frame.

## Competitor table

| Work | Venue | Input | Representation | Oriented? | Occlusion‑aware? | RoPE/binding | Video? | Code |
|---|---|---|---|---|---|---|---|---|
| **SeeThrough3D** | CVPR'26 | 3D boxes + cam + text | translucent‑box **render → VAE → OSCR tokens** (FLUX mmDiT); 1 token/loc, overlaps **alpha‑blended** | ✅ (face‑color) | ✅ (alpha) | ✅ 2D (FLUX RoPE, cond ids = image grid) | ❌ image | [yes](https://github.com/va1bhavagrawal/seethrough3d) |
| **OcclusionFormer** | ICML'26 | 2D layout + Z‑order | per‑instance latents → **volume‑render composite** | ❌ 2D | ✅ (Z‑order) | — | ❌ image | paper |
| **CineMaster** | SIGGRAPH'25 | 3D boxes + cam + labels | **dense depth map + 1‑ch entity mask → DiT‑ControlNet** | ❌ (future work) | ❌ front‑most | pixel‑aligned | ✅ | **no code** |
| **3DTrajMaster** | ICLR'25 | 6DoF pose seq/entity | **per‑entity pose token** (12‑num 3×4 extrinsic→Linear) → gated self‑attn (CogVideoX‑5B); ≤3 entities, null‑padded | ✅ 6DoF | ❌ | ❌ (pose tokens **excluded** from RoPE; no projection) | ✅ | [yes](https://github.com/KwaiVGI/3DTrajMaster) |
| **Ctrl‑V** | TMLR'25 | 2D/3D boxes | boxes **painted to RGB pixels** (paint‑over, no depth sort) → VAE → ControlNet (SVD) | rendered | ❌ lossy | none | ✅ | [yes](https://github.com/oooolga/ctrl-v) |
| **MagicDrive(/-V2)** | ICLR'24 | 3D box + BEV map + cam | box 8‑corners→Fourier→**token (cross‑attn)** + dense BEV map | partial | ❌ | ❌ Fourier abs | ✅(V2) | [yes](https://github.com/cure-lab/MagicDrive) |
| **TrackDiffusion** | WACV'25 | 2D xyxy tracklets | xyxy→Fourier→**GLIGEN gated self‑attn** token | ❌ | ❌ | ❌ | ✅ | [yes](https://github.com/pixeli99/TrackDiffusion) |
| **Boximator** | 2024 | 2D box traj | Fourier coords + id + flag → token | ❌ | ❌ | ❌ | ✅ | no code |

## Code‑verified mechanism notes

- **SeeThrough3D** (`va1bhavagrawal/seethrough3d`, base = FLUX): OSCR = translucent boxes
  (`blender_backend.py`: Principled BSDF `blend_method='BLEND'`, `face_opacity=0.025`, RGBA, face
  color = orientation) rendered to **one image**, overlaps **alpha‑blended before VAE**. Cond tokens
  get FLUX RoPE (`transformer_flux.py:259,481` `FluxPosEmbed(axes_dim=(16,56,56))`) and are
  **spatially aligned to the image grid** (`pipeline.py:77‑78` cond ids = `(i*scale_h, j*scale_w)`).
  ⇒ **"shared‑frame RoPE binding" is NOT novel.** Genuine gap: FLUX has a near‑unused 1st RoPE axis;
  Wan does not (see ARCHITECTURE.md). No depth, no multi‑token stack.
- **3DTrajMaster** (`KwaiVGI/3DTrajMaster`): pose tokens explicitly **excluded from RoPE**
  (`attention_processor.py`), no projection/pixel grounding, ≤3 entities. Injector = gated
  self‑attn into every other CogVideoX‑5B block, first ~15 steps only. Frame‑local (per‑frame
  pose↔visual pairing) — good template for how to keep per‑frame conditioning cheap.
- **Ctrl‑V** (`oooolga/ctrl-v`): `_draw_bbox` paints boxes onto a black canvas (no depth sort →
  overlaps overwrite), VAE‑encoded, **added** (not concat) to SVD latents in a ControlNet.
- **MagicDrive** (`cure-lab/MagicDrive`): `bbox_embedder.py` Fourier(8 corners) ⊕ CLIP class →
  token, `torch.cat` into cross‑attn KV. Dense BEV map is a separate ControlNet branch.
- **TrackDiffusion**: `PositionNet` + `FourierEmbedder`, box = 4 numbers (xyxy), GLIGEN
  `GatedSelfAttention`.

## Honest verdict

- "occlusion‑aware oriented‑3D‑box conditioning" — **partly scooped in images** (SeeThrough3D).
- "variable‑length conditioning tokens" — **not unique** (GLIGEN/MagicDrive family).
- "RoPE / shared‑frame binding" — **not unique** (FLUX/SeeThrough3D).
- **Genuinely open:** the combination **{video + per‑pixel multi‑depth occlusion *stack* + content‑
  adaptive budget + oriented BVH boxes + per‑object appearance latent + LLM‑authoring}**. No single
  work covers it; but each piece has neighbors, so the contribution must be carried by **execution +
  a dense‑occlusion video benchmark**, not the idea alone.
