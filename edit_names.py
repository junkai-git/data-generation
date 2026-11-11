"""
Replace the recipient name on certificates with an AI text-editing model
(AnyText), to build tampered training data with ground-truth masks.

Input is the fields.jsonl written by extract_fields.py (name text + box).
For each certificate and each new name:
  1. Crop a window around the name (AnyText works at <= 768 px)
  2. Ask AnyText to write the new name inside the name's box, in the same style
  3. Resize the result back and paste ONLY the name region into the original
     (diffusion slightly alters everything it touches; the rest must stay clean)
  4. Ground-truth mask = pixels that actually changed
  5. Save image + mask + one metadata line (same layout as replace_name.py)

Two steps, because AnyText and PaddleOCR need different Python environments:

    # 1. generate: run inside the AnyText environment
    python edit_names.py generate --fields fields.jsonl --images <cert folder> \
        --output_dir tampered --anytext_dir C:/path/to/AnyText --n_per_image 2

    # 2. verify: run inside .venv (PaddleOCR); OCRs each edit and checks the
    #    new name is really there, writing metadata_verified.jsonl
    python edit_names.py verify --output_dir tampered

Use `generate --dry_run` (no AnyText needed) to check the crops and edit regions.
"""

import argparse
import json
import os
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

ANYTEXT_MODEL = "damo/cv_anytext_text_generation_editing"
MAX_CHARS = 20  # AnyText truncates longer lines
DEFAULT_PROMPT = 'a university degree certificate with the name "{name}" printed on it'
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
        sys.exit("no usable names (letters, spaces, ' and - only, max 20 characters)")
    return names


def pick_name(old, pool, rng, partial_line):
    """A different name of similar length, in the same letter case as the original."""
    n = len(old)
    # A name sharing its line with other words can't grow into them
    hi = n * (1.05 if partial_line else 1.25)
    cands = [p for p in pool if 0.75 * n <= len(p) <= hi and p.lower() != old.lower()]
    new = rng.choice(cands or pool)
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


def edit_region(box, img_shape, pad_frac=0.12):
    x0, y0, x1, y1 = box
    p = int((y1 - y0) * pad_frac)
    h, w = img_shape[:2]
    return [max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)]


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


def model_size(cw, ch, max_side):
    """Both sides multiples of 64 and <= max_side, as AnyText expects."""
    s = min(1.0, max_side / max(cw, ch))
    return max(64, round(cw * s / 64) * 64), max(64, round(ch * s / 64) * 64)


# ---------------------------------------------------------------- AnyText
def load_anytext(args):
    # The model's wrapper loads its font and config by relative path
    os.chdir(args.anytext_dir)
    sys.path.insert(0, str(args.anytext_dir))
    from modelscope.pipelines import pipeline
    kw = {"model_revision": args.model_revision}
    if args.fp16:
        kw["use_fp16"] = True  # newer AnyText versions only; older ones ignore it
    return pipeline("my-anytext-task", model=ANYTEXT_MODEL, **kw)


def anytext_edit(pipe, crop_bgr, region_in_crop, text, args, seed):
    """Returns the edited crop (BGR, same size as crop_bgr) or None, plus any warning."""
    ch, cw = crop_bgr.shape[:2]
    tw, th = model_size(cw, ch, args.max_side)
    sx, sy = tw / cw, th / ch
    rx0, ry0, rx1, ry1 = region_in_crop
    # Position image: white region on black, 3 channels (the wrapper reads channel 0)
    pos = np.zeros((th, tw, 3), np.uint8)
    pos[int(ry0 * sy):int(ry1 * sy), int(rx0 * sx):int(rx1 * sx)] = 255
    ori = cv2.cvtColor(cv2.resize(crop_bgr, (tw, th), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    results, code, warning, _ = pipe(
        {"prompt": args.prompt.format(name=text), "seed": seed,
         "draw_pos": pos, "ori_image": ori},
        mode="text-editing", image_count=1, ddim_steps=args.steps,
        cfg_scale=args.cfg_scale, show_debug=False)
    if code < 0 or not results:
        return None, warning
    out = cv2.cvtColor(np.asarray(results[0], np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(out, (cw, ch), interpolation=cv2.INTER_CUBIC), warning


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
    for k in ("fields", "images", "output_dir", "anytext_dir", "names_file"):
        if getattr(args, k):
            setattr(args, k, Path(getattr(args, k)).resolve())  # before any chdir
    if args.max_side > 768:
        sys.exit("--max_side must be <= 768 (AnyText resizes larger inputs itself)")
    if not args.dry_run and not args.anytext_dir:
        sys.exit("--anytext_dir is required (or use --dry_run)")

    out_dir = args.output_dir
    for sub in (("debug",) if args.dry_run else ("images", "masks")):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in open(args.fields, encoding="utf-8") if l.strip()]
    pool = None if args.dry_run else name_pool(args)
    rng = random.Random(args.seed)
    pipe = None if args.dry_run else load_anytext(args)

    meta_path = out_dir / "metadata.jsonl"
    meta = None if args.dry_run else open(meta_path, "a", encoding="utf-8")
    done = skipped = 0
    for i, rec in enumerate(records):
        name = rec.get("name")
        if not name or not name.get("text"):
            print(f"skip (no name): {rec['file']}")
            skipped += 1
            continue
        box = name.get("text_box")
        if box is None and name.get("partial_line"):
            print(f"skip (name shares its line; re-run extract_fields.py for text_box): {rec['file']}")
            skipped += 1
            continue
        box = box or name["box"]
        img = cv2.imread(str(args.images / rec["file"]))
        if img is None:
            print(f"skip (unreadable): {rec['file']}")
            skipped += 1
            continue

        box = snap_box(img, box)
        region = edit_region(box, img.shape)
        cx0, cy0, cx1, cy1 = crop_window(region, img.shape)
        crop = img[cy0:cy1, cx0:cx1]
        r_in = [region[0] - cx0, region[1] - cy0, region[2] - cx0, region[3] - cy0]
        stem = Path(rec["file"]).stem

        if args.dry_run:
            dbg = crop.copy()
            cv2.rectangle(dbg, tuple(r_in[:2]), tuple(r_in[2:]), (0, 0, 255), 2)
            tw, th = model_size(cx1 - cx0, cy1 - cy0, args.max_side)
            cv2.imwrite(str(out_dir / "debug" / f"{stem}_crop.png"), dbg)
            print(f"dry run: {rec['file']} name={name['text']!r} crop={cx1 - cx0}x{cy1 - cy0} "
                  f"-> model {tw}x{th}, edit region {r_in}")
            done += 1
            continue

        for k in range(args.n_per_image):
            new = pick_name(name["text"], pool, rng, name.get("partial_line", False))
            seed = args.seed + i * 1000 + k
            edited, warning = anytext_edit(pipe, crop, r_in, new, args, seed)
            if edited is None:
                print(f"fail: {rec['file']} -> {new!r}: {warning}")
                continue
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
                "field": "name", "method": "anytext", "old_text": name["text"],
                "new_text": new, "box": box, "edit_region": region,
                "crop": [cx0, cy0, cx1, cy1], "prompt": args.prompt.format(name=new),
                "seed": seed, "steps": args.steps, "cfg_scale": args.cfg_scale,
                "model_revision": args.model_revision, "warning": warning or None,
                "changed_pixels": int((mask > 0).sum()),
            }, ensure_ascii=False) + "\n")
            meta.flush()
            print(f"done: {rec['file']} {name['text']!r} -> {new!r}")
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

    g = sub.add_parser("generate", help="Edit names with AnyText (AnyText environment)")
    g.add_argument("--fields", required=True, help="fields.jsonl from extract_fields.py")
    g.add_argument("--images", required=True, help="Folder with the certificate images")
    g.add_argument("--output_dir", required=True)
    g.add_argument("--anytext_dir", help="Path to the cloned AnyText repository")
    g.add_argument("--n_per_image", type=int, default=1)
    g.add_argument("--names_file", help="One name per line; default uses Faker")
    g.add_argument("--prompt", default=DEFAULT_PROMPT, help="Must contain {name}")
    g.add_argument("--steps", type=int, default=20)
    g.add_argument("--cfg_scale", type=float, default=9.0)
    g.add_argument("--max_side", type=int, default=768)
    g.add_argument("--diff_thresh", type=int, default=20,
                   help="Pixel change (0-255) counted as tampered in the mask")
    g.add_argument("--model_revision", default="v1.1.3")
    g.add_argument("--fp16", action="store_true", help="Half precision, if your AnyText version supports it")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--dry_run", action="store_true",
                   help="No AnyText: save crops with the edit region drawn, to check placement")

    v = sub.add_parser("verify", help="OCR each edit to check the new name (.venv)")
    v.add_argument("--output_dir", required=True)
    v.add_argument("--min_ratio", type=float, default=0.8,
                   help="Similarity between wanted and OCR-read name to count as a pass")

    args = ap.parse_args()
    generate(args) if args.cmd == "generate" else verify(args)


if __name__ == "__main__":
    main()
