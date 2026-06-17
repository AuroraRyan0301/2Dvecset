# OBB‑Driven Video Generation — Design

> Task name: **`obb`** (implemented as `wan/modules/obb`, registered as `obb-5B`). Drive video
> generation from a sparse set of **oriented 3D bounding boxes (OBBs)**, organized as a **BVH**,
> projected by a **known camera**, and injected into the pretrained **Wan2.2** DiT (verified on
> TI2V-5B) as a **content‑adaptive, variable‑length, occlusion‑preserving token stack**.
> Implementation/structure details: see `ARCHITECTURE.md` and `wan/modules/obb/README.md`.

---

## 1. One‑paragraph thesis

3D generation / controllable video sits on an axis of **control bandwidth vs. authorability**.
A full per‑pixel **G‑buffer** (à la NVIDIA *diffusion‑renderer*) is maximally informative but
**uncontrollable** — you'd have to author every pixel. A **text prompt** is authorable but too
low‑bandwidth for precise multi‑object spatio‑temporal control. **A set of oriented 3D boxes is the
sweet spot**: enough 3D structure to control layout / motion / identity, compact enough that a
human, a game engine, or an **LLM** can author it. We treat the OBB set as a **low‑dimensional,
LLM‑authorable proxy for a coarse G‑buffer**: the known camera projects it for free into a coarse
geometric scaffold (footprint, per‑pixel box‑surface depth, occlusion ordering); the video model
imagines all fine detail inside. Material / lighting are deliberately **out of scope for the proxy**
(left to a bolt‑on ControlNet if needed).

## 2. What the OBB carries (the proxy)

Per object (the part you author), all low‑dimensional:

| field | dim | note |
|---|---|---|
| center | 3 | world coords |
| size (extent) | 3 | log‑encoded |
| rotation | 6 | 6D continuous rotation rep (→ 3×3) |
| class id | 1 | categorical → embedding |
| instance id | 1 | categorical → embedding (identity tag) |
| appearance latent | ~16–64 | "what it looks like" code (ref image / 3D asset encoder) |
| bvh level | 1 | tree depth → embedding |
| static/dynamic flag | 1 | static objects cached/shared across frames |
| per‑frame trajectory | — | center+rotation sampled per (key)frame |

Global: known camera trajectory (per‑frame extrinsics + intrinsics). ~50 objects max in practice
(500 was an extreme example). Whole‑scene proxy ≈ a few thousand numbers vs. millions for a dense
G‑buffer — **three orders of magnitude smaller**, and LLM‑writable.

**We never specify the object's true shape/silhouette inside the box.** The box is a loose
spacetime envelope; the model imagines the (possibly deforming, "4D") content. The box only
constrains the *rigid* trajectory; non‑rigid articulation is the model's job.

## 3. The representation (core contribution)

Project each OBB through the known camera. For every **latent‑grid pixel** the box's footprint
covers, emit **one token**, carrying that pixel's **projected depth** (entry depth of the
ray–OBB intersection) plus the object's id / appearance / level. Key properties:

1. **Occlusion‑preserving stack.** A pixel covered by N overlapping boxes yields **N tokens**
   (one per box, each with its own depth) — *not* a single front‑most value. Occluded geometry is
   never destroyed at the conditioning stage. This is the original motivation and the headline
   property.
2. **Content‑adaptive, variable length, per frame.** Token count = Σ(footprint coverage × overlap
   depth), so it **scales with geometric complexity** and varies per frame. Sparse scenes → few
   tokens; dense/occluded scenes → more. Supports **test‑time token scaling** (LATTICE/VoxSet
   lineage).
3. **Shared coordinate frame + RoPE.** Each token's projected `(t, u, v)` is in the *same* frame as
   the video latent patches, so the existing 3D RoPE binds a patch to spatially‑nearby box tokens
   geometrically (not learned‑from‑scratch). Depth is carried as a **feature/bias** (see
   ARCHITECTURE.md §RoPE — Wan's RoPE has no spare axis for a 4th depth dim).

> **Framing discipline (do NOT over‑claim):** the novelty is **"content‑adaptive token budget
> allocated by projected coverage × occlusion depth, as an occlusion stack, on video"** — *not*
> "variable‑length tokens" (GLIGEN/MagicDrive already have variable, padded per‑object token sets)
> and *not* "we introduce RoPE/shared‑frame binding" (FLUX/SeeThrough3D already RoPE‑align
> condition tokens to the image grid). See RELATED_WORK.md.

## 4. Paradigm decision (settled)

This is **(A) a permanent OBB control interface** — boxes present at both train and test — **not
(B)** injecting a droppable 3D prior. At test the boxes can be authored by a human, a 3D engine, or
an **LLM emitting a 3D "screenplay"** (objects, poses, trajectories over time). 3D boxes are the
one interface an LLM can actually author at precision; dense depth/seg maps are not. This separates
us from text‑to‑video (too coarse) and 2D‑ControlNet video (needs hand‑drawn dense maps).

Open fork — the LLM's box "vocabulary": (i) discretized metric poses, (ii) relational/symbolic
("A left of B, approaching") compiled to metric by a layout module, (iii) hybrid. Decide later.

## 5. Novelty deltas vs. the field (after code‑level de‑risk)

The conceptual idea of **occlusion‑aware oriented‑3D‑box conditioning is partly already taken in the
IMAGE domain** by **SeeThrough3D (CVPR'26)** and **OcclusionFormer (ICML'26)**. Our defensible,
verified deltas (ranked):

1. **Video / 4D + temporal** — both threats are image‑only. Biggest clean delta.
2. **Occlusion *stack* (multi‑depth‑token per pixel)** vs. SeeThrough3D's single alpha‑blended
   render and OcclusionFormer's per‑instance volume compositing. Only clearly wins in **dense
   overlap** → must be proven empirically.
3. **Depth‑aware binding + BVH + per‑object appearance latent + LLM‑authoring.**

Story = "the **video** version of SeeThrough3D **with a better (stack) representation**." An
extension story, not a brand‑new paradigm. For an oral, we must **beat per‑frame‑SeeThrough3D and
CineMaster on temporal consistency + occlusion** on a **dense‑occlusion video benchmark**.

## 6. Why Wan as the backbone

Wan 2.1/2.2 14B: open, SOTA, DiT with **per‑block text cross‑attention**, **3D full self‑attention
with 3D RoPE**, adaLN. Critically, **Wan‑I2V already adds a *parallel* cross‑attention for the CLIP
image condition** — i.e., the architecture already shows the canonical "add a new conditioning
modality = add a parallel cross‑attention" pattern. Our OBB control slots in as a **third parallel
cross‑attention**. It is also the backbone of the chunk/autoregressive long‑video methods
(CausVid / Self‑Forcing / Rolling‑Forcing), so long, per‑frame‑varying‑OBB videos are reachable.
See ARCHITECTURE.md.

## 7. Experiment plan (north star = occlusion + temporal)

- **Data:** Blender / Infinigen‑Sim synthetic scenes with GT OBBs + camera + GT instance/depth
  masks (free, no labeling). Condition on **loose boxes** (all we have at test); **supervise** with
  the **free GT masks** (self‑tracking / containment) — *condition coarse, supervise rich*.
- **Minimal first experiment:** simplest chain — projected `(t,u,v)` shared‑RoPE cross‑attn +
  self‑tracking on ~50 dynamic boxes, **no** appearance latent, **no** BVH → measure
  box‑alignment / instance‑IoU. Validate binding cheaply before adding complexity.
- **Then:** add appearance latent → scale boxes → add BVH hierarchy → chunk for long video.
- **Headline benchmark:** dense‑occlusion video scenarios (objects cross behind each other and
  re‑emerge); metric = identity/position consistency of occluded‑then‑revealed objects. Baselines:
  per‑frame SeeThrough3D, CineMaster (reimpl), dense‑depth ControlNet.

## 8. Open questions / risks

- Does the per‑pixel **stack** actually beat SeeThrough3D's alpha‑blend render outside dense
  overlap? (If not, complexity unjustified — pick the battleground accordingly.)
- Wan RoPE has **no spare axis** for depth → depth must be a feature/bias, or RoPE re‑partitioned
  (needs FT). Validate which works.
- Cross‑chunk **appearance consistency** for long video (appearance latent + history conditioning).
- LLM box vocabulary precision (LLMs are bad at raw float poses).

See `RELATED_WORK.md` (competitor map + code‑verified mechanisms) and `ARCHITECTURE.md`
(Wan integration, injection module, RoPE, token budget).
