#!/usr/bin/env python
"""Rebuild the two figures in the README.

    uv run python assets/make_figures.py

`species_gallery.jpg` needs prepared images (`data/images_h448/`); the schematic needs
only the metadata. Both are checked in, so this only has to run when a figure changes.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch, Polygon,
                                Rectangle)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw, ImageFont

BLUE, ORANGE, RED, GREY = "#3b6ea5", "#e08214", "#c0392b", "#8a8f94"
INK, FRAME = "#1b1b1b", "#6f7479"
DPI, W, H = 190, 116, 64

# Three animal emoji, rasterised once each: matplotlib cannot draw a colour emoji itself.
_font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109)


def _emoji(codepoint: str) -> np.ndarray:
    im = Image.new("RGBA", (144, 144), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((8, 8), codepoint, font=_font, embedded_color=True)
    return np.asarray(im.crop(im.getbbox()))


ANIMALS = [_emoji("\U0001F412"), _emoji("\U0001F418"), _emoji("\U0001F993")]  # monkey, elephant, zebra

fig, ax = plt.subplots(figsize=(11.5, 6.35), dpi=DPI)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
PX_PER_UNIT = fig.get_size_inches()[0] * DPI / W


# One scene per camera: (sky, ground, [(kind, x, size), ...]). x is a fraction of the
# frame width. The background is what belongs to the camera rather than to the animal,
# so no two cameras here get the same one -- and a test camera's empty frames get the
# same scene as its labelled ones, because they come from that very camera.
SCENES = {
    "a": ("#d7e7f5", "#93ac78", [("tree", 0.16, 1.0), ("tree", 0.80, 0.7), ("bush", 0.50, 0.5)]),
    "b": ("#dfeaf5", "#d3c08d", [("mountain", 0.30, 1.0), ("mountain", 0.68, 0.7),
                                 ("bush", 0.88, 0.4)]),
    "c": ("#e7f0f7", "#c2b58e", [("sun", 0.78, 0.5), ("bush", 0.22, 0.6), ("bush", 0.44, 0.4)]),
    "d": ("#cfe0ef", "#8ea86f", [("tree", 0.34, 0.8), ("mountain", 0.74, 0.6)]),
    "e": ("#eae3d3", "#cdb887", [("tree", 0.12, 0.6), ("bush", 0.56, 0.5), ("tree", 0.86, 0.9)]),
    "f": ("#dceaf2", "#a9b98a", [("mountain", 0.20, 0.8), ("tree", 0.62, 1.0)]),
    "g": ("#d3e3ef", "#9a8f6e", [("tree", 0.24, 1.0), ("tree", 0.44, 0.6), ("tree", 0.72, 0.85)]),
    "h": ("#525a61", "#7b8288", [("moon", 0.24, 0.4), ("tree", 0.66, 0.9),
                                 ("bush", 0.40, 0.5)]),   # infrared, at night
    "i": ("#e3ecf3", "#c8bb92", [("mountain", 0.46, 1.0), ("bush", 0.14, 0.45),
                                 ("bush", 0.82, 0.55)]),
}


def scene(cx, cy, w, h, key, frame):
    """Paint one camera's background inside its frame: sky, ground, and a few features."""
    sky, ground, features = SCENES[key]
    dark = key == "h"
    horizon = cy - h / 2 + 0.38 * h
    for patch in (Rectangle((cx - w / 2, horizon), w, cy + h / 2 - horizon, facecolor=sky,
                            edgecolor="none", zorder=5.1),
                  Rectangle((cx - w / 2, cy - h / 2), w, horizon - (cy - h / 2),
                            facecolor=ground, edgecolor="none", zorder=5.1)):
        patch.set_clip_path(frame)
        ax.add_patch(patch)
    for kind, fx, size in features:
        x = cx - w / 2 + fx * w
        if kind == "tree":
            trunk = Rectangle((x - 0.035 * w, horizon - 0.02 * h), 0.07 * w, 0.30 * h * size,
                              facecolor="#6b5a45" if not dark else "#4b5157",
                              edgecolor="none", zorder=5.2)
            canopy = Circle((x, horizon + 0.30 * h * size), 0.115 * w * (0.6 + 0.4 * size),
                            facecolor="#4e7a45" if not dark else "#3c4247",
                            edgecolor="none", zorder=5.3)
            parts = (trunk, canopy)
        elif kind == "mountain":
            parts = (Polygon([[x - 0.30 * w * size, horizon],
                              [x, horizon + 0.42 * h * size],
                              [x + 0.30 * w * size, horizon]],
                             closed=True, facecolor="#8fa1ad" if not dark else "#464c52",
                             edgecolor="none", zorder=5.2),)
        elif kind == "bush":
            parts = (Circle((x, horizon + 0.03 * h), 0.07 * w * (0.8 + size),
                            facecolor="#6f8a5e" if not dark else "#434a50",
                            edgecolor="none", zorder=5.3),)
        else:                                   # sun or moon
            parts = (Circle((x, cy + h * 0.30), 0.055 * w * (0.8 + size),
                            facecolor="#f0d67a" if kind == "sun" else "#e8ecef",
                            edgecolor="none", zorder=5.2),)
        for part in parts:
            part.set_clip_path(frame)
            ax.add_patch(part)


def photo(cx, cy, key, animal=0, w=7.8, h=5.6):
    """One stylised photograph from camera `key`, with or without an animal in it.

    Args:
        cx, cy: centre of the frame.
        key: which scene, i.e. which camera the photograph comes from.
        animal: index into `ANIMALS`, or None for an empty frame.
        w, h: frame size.
    """
    frame = Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=1.1, edgecolor=FRAME,
                      facecolor="white", zorder=5.5, fill=False)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=0,
                           facecolor="white", zorder=5.0))
    scene(cx, cy, w, h, key, frame)
    ax.add_patch(frame)
    if animal is not None:
        art = ANIMALS[animal]
        # OffsetImage sizes its image in points, so the dpi has to be divided back out.
        zoom = (h * 0.62 * PX_PER_UNIT) / art.shape[0] * 72 / DPI
        ax.add_artist(AnnotationBbox(OffsetImage(art, zoom=zoom),
                                     (cx, cy - h * 0.06), frameon=False, zorder=6))


def photos(cx, cy, keys, animals, gap=8.5):
    """A row of photographs, one per camera in `keys`."""
    n = len(keys)
    for i, (key, animal) in enumerate(zip(keys, animals)):
        photo(cx + (i - (n - 1) / 2) * gap, cy, key, animal)


def box(x, y, w, h, title, lines, colour, dashed=False, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                linewidth=1.6, edgecolor=colour, facecolor=fill,
                                linestyle="--" if dashed else "-", zorder=2))
    ax.text(x + w / 2, y + h - 4.2, title, ha="center", va="center", fontsize=10.5,
            color=colour, fontweight="bold", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 9.6 - 4.6 * i, ln, ha="center", va="center",
                fontsize=9.2, color=INK, zorder=3)


def arrow(x1, y1, x2, y2, colour=INK, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color=colour, zorder=4,
                                 linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))


box(2, 40, 29, 21, "TRAIN  ·  216 cameras", ["85,265 labelled images"], BLUE)
photos(16.5, 44.0, "abc", (0, 1, 2))
box(2, 6, 29, 21, "VAL  ·  32 cameras", ["14,652 images, model selection"], ORANGE)
photos(16.5, 10.0, "def", (2, 0, 1))

box(43, 28, 22, 16, "MODEL", ["60 species"], INK, fill="#f4f4f4")

box(81, 36, 33, 25, "TEST  ·  46 cameras", ["27,764 images", "never seen in training"],
    RED)
photos(97.5, 40.0, "ghi", (1, 2, 0))
box(81, 2, 33, 25, "UNLABELLED", ["14,106 empty frames", "from 35 of those cameras"],
    GREY, dashed=True)
# Same three cameras as the test row, same backgrounds, no animal in them.
photos(97.5, 6.0, "ghi", (None, None, None))

arrow(31.6, 49, 43, 41)
arrow(31.6, 17, 43, 31)
arrow(65.6, 37, 81, 45)
ax.text(73.3, 49.6, "evaluate", ha="center", va="center", fontsize=9, color=INK)
ax.text(97.5, 32.5, "scored by macro F1 over the species present", ha="center",
        va="center", fontsize=8.6, color=INK)

arrow(81, 14, 60, 28.4, colour=GREY, dashed=True)
ax.text(66.5, 12.6, "allowed at training time", ha="center", va="center", fontsize=8.6,
        color=GREY, style="italic")

ax.plot([76.5, 76.5], [1, 62], color=GREY, linewidth=1.1, linestyle=(0, (4, 4)), zorder=1)
ax.text(76.5, 63.2, "no labelled image crosses this line", ha="center", va="center",
        fontsize=9, color=GREY, style="italic")

fig.tight_layout(pad=0.3)
fig.savefig("assets/transfer_task.png", dpi=DPI, bbox_inches="tight", facecolor="white")
print("wrote assets/transfer_task.png")


# --- the species gallery ----------------------------------------------------------------

# Eight frames chosen by hand from daylight images that carry an animal count, so the
# animal is actually in the frame -- labels are per burst, and many frames of a labelled
# burst show nothing.
PICKS = [("8f3583ac-21bc-11ea-a13a-137349068a90.jpg", "loxodonta africana", 366),
         ("870678bc-21bc-11ea-a13a-137349068a90.jpg", "giraffa camelopardalis", 170),
         ("917fa0de-21bc-11ea-a13a-137349068a90.jpg", "equus quagga", 306),
         ("87b04f68-21bc-11ea-a13a-137349068a90.jpg", "acryllium vulturinum", 532),
         ("9951c134-21bc-11ea-a13a-137349068a90.jpg", "panthera onca", 57),
         ("945b38cc-21bc-11ea-a13a-137349068a90.jpg", "tayassu pecari", 119),
         ("8dfdca76-21bc-11ea-a13a-137349068a90.jpg", "nasua narica", 519),
         ("9804e22a-21bc-11ea-a13a-137349068a90.jpg", "macaca sp", 324)]


def species_gallery(out: str = "assets/species_gallery.jpg") -> None:
    """Write the header image: one square crop per species, captioned."""
    from iwildcam.data import image_dir_for

    images = image_dir_for(448)
    cols, tile, gap, bar = 4, 420, 6, 40
    rows = (len(PICKS) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile + (cols - 1) * gap,
                              rows * (tile + bar) + (rows - 1) * gap), "white")
    draw = ImageDraw.Draw(sheet)
    name_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    cam_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    for i, (name, species, camera) in enumerate(PICKS):
        r, c = divmod(i, cols)
        im = Image.open(images / name).convert("RGB")
        scale = max(tile / im.width, tile / im.height)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        left, top = (im.width - tile) // 2, (im.height - tile) // 2
        x, y = c * (tile + gap), r * (tile + bar + gap)
        sheet.paste(im.crop((left, top, left + tile, top + tile)), (x, y))
        draw.text((x + 2, y + tile + 4), species, fill="black", font=name_font)
        draw.text((x + 2, y + tile + 22), f"camera {camera}", fill=(110, 110, 110),
                  font=cam_font)
    sheet.save(out, quality=88, optimize=True)
    print("wrote", out)


species_gallery()
