"""Publication-style architecture figure for the OBB->Wan2.2 conditioning.

Color = token/latent source; fill tint = frozen (Wan) vs trainable (OBB). Vector PDF + PNG.
Run:  PYTHONPATH=$PWD <py> scripts/make_arch_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ---- palette (by source) ----
C = dict(video="#8a8f98", text="#3b78c3", obb="#e8893b", cam="#8e5bbf",
         neutral="#eceff3", frozen="#dfe3e8", ink="#2b2b2b")


def tint(hexc, a=0.30):
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return (1 - a + a * r / 255, 1 - a + a * g / 255, 1 - a + a * b / 255)


B = {}  # name -> (cx, cy, w, h)


def box(ax, name, cx, cy, w, h, text, fc, ec, fs=8.5, bold=False, lw=1.6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.015,rounding_size=0.10", fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=C["ink"],
            fontweight="bold" if bold else "normal", zorder=4)
    B[name] = (cx, cy, w, h)


def side(name, s):
    cx, cy, w, h = B[name]
    return {"r": (cx + w / 2, cy), "l": (cx - w / 2, cy), "t": (cx, cy + h / 2),
            "b": (cx, cy - h / 2)}[s]


def arr(ax, p1, p2, color, ls="-", lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, color=color,
                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=2,
                 shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(15, 8.4))
ax.set_xlim(0, 15); ax.set_ylim(0, 8.4); ax.axis("off")

# ===== (a) Condition generation (trainable OBB conditioner) =====
ax.text(0.2, 8.15, "(a) OBB Conditioner  (trainable; no scene encoder)", fontsize=11, fontweight="bold", color=C["ink"])
box(ax, "inp", 2.0, 7.1, 3.1, 1.05, "Scene: N oriented OBBs\n{center(t), R(t), size, appear?}\n+ known Camera {K, R_cw, t_cw}", C["neutral"], "#9aa0a8")
box(ax, "proj", 4.95, 7.1, 1.8, 1.0, "①  Projection\nray–OBB ∩, rasterize\n@ cond_scale × latent", C["neutral"], "#9aa0a8")
box(ax, "pix", 8.05, 7.1, 3.3, 1.1, "per-pixel occlusion-stack tokens\n(flat + cu_seqlens — varlen)\nid · depth · world_normal · appear\npos(t,h,w)  fractional", tint(C["obb"]), C["obb"], fs=8)
box(ax, "emb", 11.4, 7.1, 2.6, 1.1, "②  OBBTokenEmbedder ★\nid(RFF) + Mellin(depth)\n+ normal + appear|null", tint(C["obb"]), C["obb"], fs=8)
box(ax, "obbemb", 13.85, 7.1, 1.7, 0.7, "obb_emb\n(K/V)", tint(C["obb"], .5), C["obb"], bold=True)
# camera chain
box(ax, "pluck", 4.95, 5.5, 1.8, 0.8, "build_plucker\nper-pixel rays (6-D)", C["neutral"], "#9aa0a8")
box(ax, "camenc", 8.05, 5.5, 2.2, 0.8, "CameraEncoder ★\n(zero-init)", tint(C["cam"]), C["cam"])
box(ax, "cam", 11.0, 5.5, 1.8, 0.7, "cam tokens\n(B,S,dim)", tint(C["cam"], .5), C["cam"], bold=True)

arr(ax, side("inp", "r"), side("proj", "l"), "#9aa0a8")
arr(ax, side("proj", "r"), side("pix", "l"), C["obb"])
arr(ax, side("pix", "r"), side("emb", "l"), C["obb"])
arr(ax, side("emb", "r"), side("obbemb", "l"), C["obb"])
arr(ax, side("inp", "b"), side("pluck", "l"), C["cam"], rad=0.2)
arr(ax, side("pluck", "r"), side("camenc", "l"), C["cam"])
arr(ax, side("camenc", "r"), side("cam", "l"), C["cam"])

# ===== (b) Frozen Wan2.2 backbone + injection =====
ax.text(0.2, 3.55, "(b) Frozen Wan2.2 backbone  +  injection", fontsize=11, fontweight="bold", color=C["ink"])
yb = 2.2
box(ax, "vlat", 1.4, yb, 2.0, 0.95, "video latent\n[C,F,H,W] (noisy)", tint(C["video"], .25), C["video"])
box(ax, "pe", 3.5, yb, 1.5, 0.95, "patch_embed\n❄", C["frozen"], "#9aa0a8")
ax.add_patch(Circle((4.95, yb), 0.22, fc="white", ec=C["cam"], lw=1.8, zorder=3)); ax.text(4.95, yb, "+", ha="center", va="center", fontsize=13, color=C["cam"], zorder=4)
bx = [("b0", 6.2, "Block 0"), ("b1", 7.7, "Block 1"), ("b2", 9.2, "Block 2")]
for n, cx, lab in bx:
    box(ax, n, cx, yb, 1.25, 1.05, lab + "\n❄", C["frozen"], "#9aa0a8")
ax.text(10.4, yb, "· · ·", fontsize=14, ha="center", va="center", color=C["ink"])
box(ax, "head", 11.5, yb, 1.2, 1.05, "head\n❄", C["frozen"], "#9aa0a8")
box(ax, "vout", 13.4, yb, 1.7, 0.95, "velocity v\n[C,F,H,W]", tint(C["video"], .25), C["video"], bold=True)

arr(ax, side("vlat", "r"), side("pe", "l"), C["video"])
arr(ax, side("pe", "r"), (4.73, yb), C["video"])
arr(ax, (5.17, yb), side("b0", "l"), C["video"])
arr(ax, side("b0", "r"), side("b1", "l"), C["video"])
arr(ax, side("b1", "r"), side("b2", "l"), C["video"])
arr(ax, side("b2", "r"), (9.95, yb), C["video"])
arr(ax, (10.85, yb), side("head", "l"), C["video"])
arr(ax, side("head", "r"), side("vout", "l"), C["video"])
# camera added once
arr(ax, side("cam", "b"), (4.95, yb + 0.22), C["cam"], rad=-0.1)
ax.text(5.15, 4.25, "camera added ONCE", fontsize=7.5, color=C["cam"], style="italic")
# OBB cross-attn injected at even blocks only
for n, cx, _ in [bx[0], bx[2]]:
    arr(ax, side("obbemb", "b"), (cx, yb + 0.55), C["obb"], ls=(0, (4, 2)), rad=0.0)
    ax.add_patch(Circle((cx, yb + 0.55), 0.13, fc="white", ec=C["obb"], lw=1.6, zorder=5)); ax.text(cx, yb + 0.55, "+", ha="center", va="center", fontsize=9, color=C["obb"], zorder=6)
ax.text(6.2, yb + 1.06, "OBB cross-attn (flash varlen, gated, zero-init)\nQ=video (integer pos)  K/V=obb_emb (fractional pos)  shared (t,h,w) RoPE", fontsize=7.0, ha="center", color=C["obb"])
ax.text(7.7, yb + 0.95, "no inject", fontsize=7, ha="center", color="#9aa0a8", style="italic")
ax.text(8.0, yb - 0.85, "inject only at obb_inject_layers (default every-other block — NOT every block)",
        fontsize=8, color=C["obb"])
# base Wan conditions: text + timestep (every block)
box(ax, "txt", 2.0, 1.15, 2.3, 0.5, "text (umT5)  +  t", tint(C["text"]), C["text"], fs=8)
arr(ax, side("txt", "r"), side("b0", "b"), C["text"], rad=-0.12)
ax.text(4.2, 1.5, "text→cross-attn,  t→adaLN  (every block)", fontsize=7, color=C["text"], style="italic")

# ===== legend =====
lx, ly = 0.3, 0.55
items = [("video latent", C["video"]), ("text (umT5)", C["text"]), ("OBB tokens", C["obb"]),
         ("camera (Plücker)", C["cam"]), ("❄ frozen Wan", C["frozen"]), ("★ trainable", "white")]
for i, (lab, col) in enumerate(items):
    x = lx + i * 2.4
    ax.add_patch(FancyBboxPatch((x, ly), 0.28, 0.18, boxstyle="round,pad=0.01", fc=col, ec="#888", lw=1.2))
    ax.text(x + 0.36, ly + 0.09, lab, fontsize=7.6, va="center", color=C["ink"])

os.makedirs("docs/figs", exist_ok=True)
fig.tight_layout()
fig.savefig("docs/figs/arch.pdf", bbox_inches="tight")
fig.savefig("docs/figs/arch.png", dpi=200, bbox_inches="tight")
print("saved docs/figs/arch.pdf and docs/figs/arch.png")
