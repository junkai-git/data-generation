"""
Draw the field boxes from fields.jsonl (written by extract_fields.py) onto the
certificate images, and save a binary mask of the same boxes for each image.

Overlays and masks go to separate folders and never next to the source images:
    <out>/overlays/C_001_clean.jpg   image with labelled boxes, for checking
    <out>/masks/C_001_clean.png      white boxes on black, same size as the image

Masks are PNG so the edges stay exactly 0/255 (JPEG would blur them), which
makes them usable directly as inpainting masks.

Example:
    python visualize_boxes.py
    python visualize_boxes.py --fields name date --pad 6
    python visualize_boxes.py --images "../dataset/Dataset Certs" --out name_boxes
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {"name": (0, 0, 255), "date": (0, 160, 0), "degree": (255, 120, 0)}  # BGR
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def index_images(root):
    """Map file name -> path for every image under root (first match wins)."""
    found = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            found.setdefault(p.name, p)
    return found


def clamp_box(box, pad, w, h):
    x0, y0, x1, y1 = box
    return (max(0, x0 - pad), max(0, y0 - pad), min(w - 1, x1 + pad), min(h - 1, y1 + pad))


def draw_label(img, text, x, y, color):
    scale = max(0.5, img.shape[1] / 2000)
    thick = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(y, th + base + 2)  # keep the label on screen for boxes at the top edge
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, default=HERE / "fields.jsonl")
    ap.add_argument("--images", type=Path, default=HERE.parent / "dataset" / "Dataset Certs",
                    help="folder searched recursively for the files named in the jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "field_boxes")
    ap.add_argument("--fields", nargs="+", default=["name"], choices=list(COLORS))
    ap.add_argument("--pad", type=int, default=0, help="grow each box by this many pixels")
    ap.add_argument("--no-text", action="store_true", help="draw boxes without the field text")
    args = ap.parse_args()

    overlay_dir, mask_dir = args.out / "overlays", args.out / "masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    images = index_images(args.images)
    done = missing_img = no_box = 0

    with open(args.jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for rec in records:
        path = images.get(rec["file"])
        if path is None:
            print(f"[skip] image not found: {rec['file']}")
            missing_img += 1
            continue

        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)  # handles non-ASCII paths
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        thick = max(2, w // 600)
        drawn = []

        for field in args.fields:
            info = rec.get(field)
            if not info or not info.get("box"):
                continue
            x0, y0, x1, y1 = clamp_box(info["box"], args.pad, w, h)
            mask[y0:y1 + 1, x0:x1 + 1] = 255
            cv2.rectangle(img, (x0, y0), (x1, y1), COLORS[field], thick)
            label = field if args.no_text else f"{field}: {info.get('text', '')}"
            draw_label(img, label, x0, y0 - thick, COLORS[field])
            drawn.append(field)

        if not drawn:
            print(f"[warn] no {'/'.join(args.fields)} box: {rec['file']}")
            no_box += 1

        # Masks are written even when empty, so every image has a matching mask
        stem = Path(rec["file"]).stem
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(overlay_dir / f"{stem}.jpg"))
        cv2.imencode(".png", mask)[1].tofile(str(mask_dir / f"{stem}.png"))
        done += 1

    print(f"\n{done} written, {no_box} without a box, {missing_img} images not found")
    print(f"overlays: {overlay_dir}\nmasks:    {mask_dir}")


if __name__ == "__main__":
    main()
