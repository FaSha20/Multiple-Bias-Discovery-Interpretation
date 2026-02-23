# utils/grid_utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def chunk_list(items: Sequence, batch_size: int) -> List[List]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def _load_font(font_size: int) -> ImageFont.ImageFont:
    """
    Tries to load a nicer monospace font; falls back to default.
    """
    try:
        # Common on many linux environments
        return ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def _to_pil(img) -> Image.Image:
    """
    Accepts PIL Image, numpy array (H,W,C), or torch tensor-like with .numpy().
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if hasattr(img, "detach") and hasattr(img, "cpu"):  # torch tensor
        img = img.detach().cpu().numpy()
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] in (1, 3):
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            return Image.fromarray(img, mode="RGB")
        raise ValueError("Unsupported numpy image format; expected HxWxC")
    raise ValueError(f"Unsupported image type: {type(img)}")


def _resize_and_pad(img: Image.Image, size: int) -> Image.Image:
    """
    Resize to fit within (size,size) while preserving aspect ratio, then pad.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), color=(0, 0, 0))

    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img_r = img.resize((nw, nh), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas.paste(img_r, (ox, oy))
    return canvas


@dataclass(frozen=True)
class GridBuildResult:
    grid_image: Image.Image
    tile_id_to_index: Dict[str, int]
    tile_ids: List[str]


def build_labeled_grid(
    images: Sequence,
    indices: Sequence[int],
    rows: int,
    cols: int,
    tile_size: int,
    pad: int,
    font_size: int,
    tile_id_prefix: str = "img",
) -> GridBuildResult:
    """
    Build a rows x cols grid (or fewer tiles if images < rows*cols),
    overlaying each tile with a unique ID, and returning mapping tile_id -> dataset index.
    """
    if len(images) != len(indices):
        raise ValueError("images and indices must have same length")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be > 0")

    n_slots = rows * cols
    n = min(len(images), n_slots)

    font = _load_font(font_size)

    grid_w = cols * tile_size + (cols + 1) * pad
    grid_h = rows * tile_size + (rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid)

    tile_id_to_index: Dict[str, int] = {}
    tile_ids: List[str] = []

    for t in range(n):
        r = t // cols
        c = t % cols
        x0 = pad + c * (tile_size + pad)
        y0 = pad + r * (tile_size + pad)

        pil = _to_pil(images[t])
        tile = _resize_and_pad(pil, tile_size)
        grid.paste(tile, (x0, y0))

        tile_id = f"{tile_id_prefix}_{t:04d}"
        tile_id_to_index[tile_id] = int(indices[t])
        tile_ids.append(tile_id)

        # draw label box
        label = tile_id
        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        box_pad = 3
        bx0 = x0 + 3
        by0 = y0 + 3
        bx1 = bx0 + text_w + 2 * box_pad
        by1 = by0 + text_h + 2 * box_pad

        draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0))
        draw.text((bx0 + box_pad, by0 + box_pad), label, fill=(255, 255, 255), font=font)

    return GridBuildResult(grid_image=grid, tile_id_to_index=tile_id_to_index, tile_ids=tile_ids)


def save_grid_image(grid: Image.Image, out_path: str | Path, quality: int = 95) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        grid.save(out_path, quality=quality)
    else:
        grid.save(out_path)
    return out_path