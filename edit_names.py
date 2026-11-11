"""
Replace the recipient name on certificates with a locally hosted AI image
editor (Qwen-Image-Edit-2511) that copies the original name's lettering, to
build tampered training data with ground-truth masks.

Input is the fields.jsonl written by extract_fields.py (name text + box).
For each certificate and each new name:
  1. Crop a window around the name. The model sees the original name, so it
     can copy its font, size, colour and ink texture.
  2. Ask the model to change the old name to the new one, keeping the style
  3. Undo any small shift the model introduced, then paste ONLY the name
     region into the original (the model slightly redraws everything it sees;
     the rest of the certificate must stay untouched)
  4. Ground-truth mask = pixels that actually changed
  5. Save image + mask + one metadata line

    python edit_names.py generate --fields fields.jsonl --images <cert folder> \
        --output_dir tampered --n_per_image 2
    python edit_names.py verify --output_dir tampered

Everything runs locally in the same .venv as extract_fields.py. The model
(about 20B parameters) downloads once from Hugging Face. Use
`generate --dry_run` to check the crops without loading it.
"""

import argparse
import json
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_PROMPT = ('Change the name "{old}" to "{new}". Keep exactly the same font, '
                  'letter style, size, weight, colour, spacing and position. '
                  'Do not change anything else in the image.')
MAX_CHARS = 30
NAME_OK = re.compile(r"^[A-Za-z][A-Za-z'\- ]+$")


# ---------------------------------------------------------------- names
def name_pool(args):
    if args.names_file:
        names = [l.strip() for l in open(args.names_file, encoding="utf-8") if l.strip()]
    else:
        from faker import Faker
        Faker.seed(args.seed)
        fk = Faker(["en_US", "en_GB", "en_IN"])
        names = list({fk.name() for _ in range(20000)})
    names = sorted(n for n in names if NAME_OK.match(n) and len(n) <= MAX_CHARS)
    if not names:
        sys.exit(f"no usable names (letters, spaces, ' and - only, max {MAX_CHARS} characters)")
    return names


def pick_name(old, pool, rng, partial_line):
    """A different name of similar length, in the same letter case as the original."""
    n = len(old)
    # A name sharing its line with other words can't grow into them
    hi = n * (1.05 if partial_line else 1.25)
    others = [p for p in pool if p.lower() != old.lower()]
    cands = [p for p in others if 0.75 * n <= len(p) <= hi]
    if not cands:  # very short or long originals: take the closest lengths available
        cands = sorted(others, key=lambda p: abs(len(p) - n))[:50]
    new = rng.choice(cands)
    if old.isupper():
        return new.upper()
    if old.islower():
        return new.lower()
    return new


# ---------------------------------------------------------------- geometry
def snap_box(img, box, reach=0.8, min_gap=0.25):
    """Move the left/right edges to the nearest gap between words.

    PaddleOCR's word boxes are estimated from character positions and can cut
    into a letter or reach into the next word; gaps are found from the ink.
    """
    x0, y0, x1, y1 = box
    h = y1 - y0
    lo, hi = max(0, x0 - int(reach * h)), min(img.shape[1], x1 + int(reach * h))
    strip = cv2.cvtColor(img[y0:y1, lo:hi], cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(strip, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = list(dark.sum(axis=0) > max(1, 0.04 * h))
    gaps, start = [], None  # ink-free column runs wide enough to be word gaps
    for i, v in enumerate(ink + [True]):
        if not v and start is None:
            start = i
        elif v and start is not None:
            if i - start >= min_gap * h or start == 0 or i == len(ink):
                gaps.append((start, i))
            start = None
    if not gaps:
        return box
    nx0 = min((e for _, e in gaps), key=lambda e: abs(e - (x0 - lo)))  # where ink starts
    nx1 = min((s for s, _ in gaps), key=lambda s: abs(s - (x1 - lo)))  # where ink stops
    return [lo + nx0, y0, lo + nx1, y1] if nx1 > nx0 else box


def edit_region(box, img_shape, partial_line, pad_frac=0.12, grow_frac=0.15):
    """The area pasted back from the model's output. A name on its own line is
    usually centred and a longer new name spreads both ways, so it gets room to
    grow sideways; a name sharing its line must not spill into its neighbours."""
    x0, y0, x1, y1 = box
    py = int((y1 - y0) * pad_frac)
    px = py if partial_line else py + int((x1 - x0) * grow_frac)
    h, w = img_shape[:2]
    return [max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)]


def crop_window(region, img_shape, max_ratio=2.0):
    """Region plus a line height of context, made no wider than max_ratio:1 so
    the model sees a normal-shaped image rather than a thin strip."""
    x0, y0, x1, y1 = region
    ctx = y1 - y0
    cx0, cy0, cx1, cy1 = x0 - ctx, y0 - ctx, x1 + ctx, y1 + ctx
    cw, ch = cx1 - cx0, cy1 - cy0
    if cw > max_ratio * ch:
        extra = (cw / max_ratio - ch) / 2
        cy0, cy1 = cy0 - extra, cy1 + extra
    h, w = img_shape[:2]
    return [max(0, int(cx0)), max(0, int(cy0)), min(w, int(cx1)), min(h, int(cy1))]


# ---------------------------------------------------------------- model
def load_editor(args):
    import torch
    from diffusers import QwenImageEditPlusPipeline

    kw = {"torch_dtype": torch.bfloat16}
    if args.quantize == "4bit":
        # Community-tested route to fit 16-24 GB GPUs; not in the official docs
        from diffusers.quantizers import PipelineQuantizationConfig
        kw["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder"])
    print(f"loading {args.model} (quantize={args.quantize}, offload={args.offload})")
    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model, **kw)
    if args.offload:
        pipe.enable_model_cpu_offload()  # slower, but needs far less GPU memory
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def edit_crop(pipe, crop_bgr, old, new, args, seed):
    """Edited crop, resized back to the crop's size (the model picks its own
    working resolution)."""
    import torch
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    with torch.inference_mode():
        out = pipe(image=[image], prompt=args.prompt.format(old=old, new=new),
                   negative_prompt=" ", true_cfg_scale=args.cfg_scale,
                   num_inference_steps=args.steps, num_images_per_prompt=1,
                   generator=torch.Generator().manual_seed(seed)).images[0]
    out = cv2.cvtColor(np.asarray(out.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = crop_bgr.shape[:2]
    interp = cv2.INTER_AREA if out.shape[1] > w else cv2.INTER_CUBIC
    return cv2.resize(out, (w, h), interpolation=interp)


def align(edited, crop):
    """Undo a small overall shift between the model's output and the input, so
    the pasted region lines up with the untouched certificate around it."""
    a = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return edited, (0.0, 0.0)
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    shifted = cv2.warpAffine(edited, m, (edited.shape[1], edited.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return shifted, (round(dx, 2), round(dy, 2))


def spill_ratio(crop, edited, region_in_crop, thresh):
    """Changed pixels outside the pasted region relative to inside it. A high
    value means the model also redrew things elsewhere (e.g. moved the name),
    and part of its edit was not pasted back."""
    changed = np.abs(edited.astype(np.int16) - crop.astype(np.int16)).max(axis=2) > thresh * 2
    rx0, ry0, rx1, ry1 = region_in_crop
    inside = changed[ry0:ry1, rx0:rx1].sum()
    return round(float((changed.sum() - inside) / max(1, inside)), 3)


def composite(crop, edited, region_in_crop, feather):
    """Paste the edited name region into the crop with a soft edge."""
    rx0, ry0, rx1, ry1 = region_in_crop
    alpha = np.zeros(crop.shape[:2], np.float32)
    alpha[max(0, ry0 - feather):ry1 + feather, max(0, rx0 - feather):rx1 + feather] = 1
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1, feather / 2))[..., None]
    return (crop * (1 - alpha) + edited * alpha).round().astype(np.uint8)


def changed_mask(before, after, thresh):
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    m = (diff > thresh).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# ---------------------------------------------------------------- generate
def generate(args):
    out_dir = Path(args.output_dir)
    for sub in (("debug",) if args.dry_run else ("images", "masks")):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in open(args.fields, encoding="utf-8") if l.strip()]
    pool = None if args.dry_run else name_pool(args)
    rng = random.Random(args.seed)
    pipe = None if args.dry_run else load_editor(args)

    meta_path = out_dir / "metadata.jsonl"
    meta = None if args.dry_run else open(meta_path, "a", encoding="utf-8")
    done = skipped = 0
    for i, rec in enumerate(records):
        name = rec.get("name")
        if not name or not name.get("text"):
            print(f"skip (no name): {rec['file']}")
            skipped += 1
            continue
        if args.skip_review and name.get("needs_review"):
            print(f"skip (name flagged needs_review: {name['text']!r}): {rec['file']}")
            skipped += 1
            continue
        box = name.get("text_box")
        partial = bool(name.get("partial_line"))
        if box is None and partial:
            print(f"skip (name shares its line; re-run extract_fields.py for text_box): {rec['file']}")
            skipped += 1
            continue
        box = box or name["box"]
        img = cv2.imread(str(Path(args.images) / rec["file"]))
        if img is None:
            print(f"skip (unreadable): {rec['file']}")
            skipped += 1
            continue

        box = snap_box(img, box)
        region = edit_region(box, img.shape, partial)
        cx0, cy0, cx1, cy1 = crop_window(region, img.shape)
        crop = img[cy0:cy1, cx0:cx1]
        r_in = [region[0] - cx0, region[1] - cy0, region[2] - cx0, region[3] - cy0]
        stem = Path(rec["file"]).stem

        if args.dry_run:
            dbg = crop.copy()
            cv2.rectangle(dbg, tuple(r_in[:2]), tuple(r_in[2:]), (0, 0, 255), 2)
            cv2.imwrite(str(out_dir / "debug" / f"{stem}_crop.png"), dbg)
            print(f"dry run: {rec['file']} name={name['text']!r} crop={cx1 - cx0}x{cy1 - cy0}, "
                  f"paste region {r_in}")
            done += 1
            continue

        for k in range(args.n_per_image):
            new = pick_name(name["text"], pool, rng, partial)
            seed = args.seed + i * 1000 + k
            try:
                edited = edit_crop(pipe, crop, name["text"], new, args, seed)
            except Exception as e:  # e.g. out of GPU memory on one large crop
                print(f"fail: {rec['file']} -> {new!r}: {type(e).__name__}: {e}")
                continue
            edited, shift = align(edited, crop)
            spill = spill_ratio(crop, edited, r_in, args.diff_thresh)
            feather = max(2, (region[3] - region[1]) // 10)
            new_crop = composite(crop, edited, r_in, feather)
            out = img.copy()
            out[cy0:cy1, cx0:cx1] = new_crop
            mask = np.zeros(img.shape[:2], np.uint8)
            mask[cy0:cy1, cx0:cx1] = changed_mask(crop, new_crop, args.diff_thresh)

            out_name = f"{stem}_ai{k:02d}.png"
            cv2.imwrite(str(out_dir / "images" / out_name), out)
            cv2.imwrite(str(out_dir / "masks" / f"{stem}_ai{k:02d}_mask.png"), mask)
            meta.write(json.dumps({
                "source": rec["file"], "output": out_name, "mask": f"{stem}_ai{k:02d}_mask.png",
                "field": "name", "method": "qwen-image-edit", "model": args.model,
                "old_text": name["text"], "new_text": new, "box": box, "edit_region": region,
                "crop": [cx0, cy0, cx1, cy1], "prompt": args.prompt.format(old=name["text"], new=new),
                "seed": seed, "steps": args.steps, "cfg_scale": args.cfg_scale,
                "shift": shift, "spill_ratio": spill, "changed_pixels": int((mask > 0).sum()),
            }, ensure_ascii=False) + "\n")
            meta.flush()
            flag = " [check: model also changed outside the name]" if spill > args.max_spill else ""
            print(f"done: {rec['file']} {name['text']!r} -> {new!r}{flag}")
            done += 1
    if meta:
        meta.close()
    print(f"\n{done} written, {skipped} skipped" + ("" if args.dry_run else f"; metadata: {meta_path}"))


# ---------------------------------------------------------------- verify
def verify(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_fields import _norm_name, load_ocr

    out_dir = Path(args.output_dir)
    rows = [json.loads(l) for l in open(out_dir / "metadata.jsonl", encoding="utf-8") if l.strip()]
    ocr = load_ocr()
    passed = 0
    for r in rows:
        img = cv2.imread(str(out_dir / "images" / r["output"]))
        x0, y0, x1, y1 = r["edit_region"]
        p = (y1 - y0) // 3
        crop = img[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p]
        read = " ".join(ocr.predict(crop)[0]["rec_texts"])
        new, got, old = _norm_name(r["new_text"]), _norm_name(read), _norm_name(r["old_text"])
        score = SequenceMatcher(None, new, got).ratio()
        old_left = len(old) >= 4 and old in got  # original name still visible
        r.update(ocr_read=read, ocr_score=round(score, 3), old_name_visible=old_left,
                 verified=(new in got or score >= args.min_ratio) and not old_left)
        passed += r["verified"]
        print(f"{'ok  ' if r['verified'] else 'FAIL'} {r['output']}: wanted {r['new_text']!r}, read {read!r}")
    with open(out_dir / "metadata_verified.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{passed}/{len(rows)} edits verified; see {out_dir / 'metadata_verified.jsonl'}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Swap names with Qwen-Image-Edit")
    g.add_argument("--fields", required=True, help="fields.jsonl from extract_fields.py")
    g.add_argument("--images", required=True, help="Folder with the certificate images")
    g.add_argument("--output_dir", required=True)
    g.add_argument("--n_per_image", type=int, default=1)
    g.add_argument("--names_file", help="One name per line; default uses Faker")
    g.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face id or local folder")
    g.add_argument("--prompt", default=DEFAULT_PROMPT, help="Must contain {old} and {new}")
    g.add_argument("--steps", type=int, default=40)
    g.add_argument("--cfg_scale", type=float, default=4.0)
    g.add_argument("--quantize", choices=["none", "4bit"], default="none",
                   help="4bit: much less GPU memory, slightly lower quality (experimental)")
    g.add_argument("--offload", action="store_true",
                   help="Keep model parts in system RAM until needed: less GPU memory, slower")
    g.add_argument("--diff_thresh", type=int, default=20,
                   help="Pixel change (0-255) counted as tampered in the mask")
    g.add_argument("--max_spill", type=float, default=0.5,
                   help="Warn when changes outside the name exceed this fraction of those inside")
    g.add_argument("--skip_review", action="store_true",
                   help="Skip names extract_fields.py flagged needs_review (may be misread "
                        "or not a name at all, e.g. a signature)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--dry_run", action="store_true",
                   help="No model: save crops with the paste region drawn, to check placement")

    v = sub.add_parser("verify", help="OCR each edit to check the new name")
    v.add_argument("--output_dir", required=True)
    v.add_argument("--min_ratio", type=float, default=0.8,
                   help="Similarity between wanted and OCR-read name to count as a pass")

    args = ap.parse_args()
    generate(args) if args.cmd == "generate" else verify(args)


if __name__ == "__main__":
    main()
