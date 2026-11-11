"""
Extract name / date / degree from certificate images.

For each image:
  1. PaddleOCR (PP-OCRv6) finds every text line with its box and confidence
  2. An LLM copies the ncdame, date and degree text out of the OCR lines; each
     answer is matched back to the OCR line(s) it came from, which gives the box
  3. The name's line crop is sent to a vision model for a second, verbatim read
     (decorative fonts often fool OCR with high confidence)
  4. Date is normalised to ISO, degree mapped to a level; one JSON line per image

The LLM is either a local Qwen2.5-VL (default, free, nothing leaves the machine)
or the OpenAI API.

Install:
    pip install paddlepaddle paddleocr dateparser opencv-python
    # provider qwen: CUDA torch + torchvision (see pytorch.org), then
    pip install transformers accelerate bitsandbytes
    # provider openai:
    pip install openai          # and set OPENAI_API_KEY

Example:
    python extract_fields.py --input certs/ --output fields.jsonl --load_in_4bit
    python extract_fields.py --input certs/ --provider openai --model gpt-4o-mini
"""

import argparse
import base64
import json
import multiprocessing
import os
import queue
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2
from PIL import Image

DEFAULT_QWEN = "Qwen/Qwen2.5-VL-3B-Instruct"

# Checked in order, highest level first, so "Master of Science" never falls
# through to a looser match. Keywords of 6+ letters also match OCR text whose
# spaces were lost ("BACHELOROFCOMMERCE").
DEGREE_LEVELS = [
    ("doctorate", ["doctor", "phd", "ph.d", "dphil", "edd"]),
    ("master", ["master", "msc", "m.sc", "mba", "mphil", "meng", "llm", "ma", "ms"]),
    ("bachelor", ["bachelor", "bsc", "b.sc", "beng", "bcom", "llb", "ba", "bs"]),
    ("associate", ["associate"]),
    ("diploma", ["diploma"]),
    ("certificate", ["certificate"]),
]

LABEL_PROMPT = """These are OCR lines from a certificate, top to bottom.
Find three fields and copy their words exactly as they appear in the lines:
- name: the recipient's name only, without surrounding words (not signatories,
  not the institution).
- date: the date the degree was awarded or the certificate issued (not a date
  of birth or admission).
- degree: the qualification awarded only, e.g. "Master of Public Administration".
OCR may misspell decorative text; copy it as it appears. A field may continue
onto the next line. Use "" if a field is absent.
Return only JSON: {"name": "...", "date": "...", "degree": "..."}
Lines:
"""

REREAD_PROMPT = """This image is a crop from a certificate. It contains the
recipient's name, possibly with other words around it. Transcribe only the
person's name, exactly as written, letter by letter. Do not correct or normalise
the spelling and do not include any other words.
Return JSON: {"text": "...", "legible": true|false}."""

# Letters (any script), spaces and the punctuation real names use
NAME_RE = re.compile(r"^[^\W\d_]+(?:[ '\-.,][^\W\d_]*)*$")


# ---------------------------------------------------------------- OCR
def ocr_device(choice="auto"):
    """'gpu:0' when the GPU build of Paddle (paddlepaddle-gpu) can see a GPU."""
    if choice != "auto":
        return "gpu:0" if choice == "gpu" else "cpu"
    import paddle
    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        return "gpu:0"
    return "cpu"


def load_ocr(lang=None, det_max_side=1600, device="auto"):
    # PaddleOCR imports modelscope -> transformers -> torch when they're installed,
    # and PyTorch's cuDNN then clashes with the GPU build of Paddle (see OCRWorker).
    # OCR never needs torch, so hide it from this process if nothing has loaded it yet.
    if "torch" not in sys.modules:
        sys.modules["torch"] = None
    device = ocr_device(device)  # imports paddle, before anything else can
    from paddleocr import PaddleOCR
    print(f"PaddleOCR on {device}")
    # oneDNN (mkldnn) crashes on Paddle 3.x for Windows with
    # "ConvertPirAttribute2RuntimeAttribute not support"; plain CPU kernels work.
    # (Only affects CPU runs.) Detection runs on a copy downscaled to
    # det_max_side; recognition still reads full-resolution crops.
    kw = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
              use_textline_orientation=False, enable_mkldnn=False, device=device,
              text_det_limit_type="max", text_det_limit_side_len=det_max_side,
              return_word_box=True)
    if lang:
        kw["lang"] = lang
    return PaddleOCR(**kw)


def ocr_lines(img_bgr, ocr):
    res = ocr.predict(img_bgr)[0]
    n = len(res["rec_texts"])
    # Per-word boxes (estimated by PaddleOCR from character positions in the line)
    words_all = res.get("text_word") or [None] * n
    wboxes_all = res.get("text_word_boxes") or [None] * n
    lines = []
    for text, conf, poly, words, wboxes in zip(res["rec_texts"], res["rec_scores"],
                                                res["rec_polys"], words_all, wboxes_all):
        if not text.strip():
            continue
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        line = {"text": text, "conf": round(float(conf), 3),
                "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]}
        if words is not None and wboxes is not None and len(words) == len(wboxes):
            line["words"] = [{"text": w, "box": [int(v) for v in b]}
                             for w, b in zip(words, wboxes) if str(w).strip()]
        lines.append(line)
    # Reading order: group boxes into rows by vertical centre, then left to right
    lines.sort(key=lambda l: l["box"][1] + l["box"][3])
    rows = []
    for l in lines:
        cy, h = (l["box"][1] + l["box"][3]) / 2, l["box"][3] - l["box"][1]
        if rows and abs(cy - rows[-1]["cy"]) < 0.5 * min(h, rows[-1]["h"]):
            rows[-1]["items"].append(l)
        else:
            rows.append({"cy": cy, "h": h, "items": [l]})
    lines = [l for r in rows for l in sorted(r["items"], key=lambda l: l["box"][0])]
    for i, l in enumerate(lines):
        l["id"] = i
    return lines


class OCRWorker:
    """PaddleOCR in its own process.

    The GPU builds of Paddle and PyTorch ship different cuDNN versions under the
    same DLL names, and one process can only load one of them: whichever library
    loads second fails ("The specified procedure could not be found"). Keeping
    Paddle in a separate process gives each its own copy.
    """

    def __init__(self, lang=None, det_max_side=1600, device="auto"):
        self.cfg = (lang, det_max_side, device)
        self._start()

    def _start(self):
        ctx = multiprocessing.get_context("spawn")
        self.req, self.res = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_ocr_worker, args=(self.req, self.res, *self.cfg),
                                daemon=True)
        self.proc.start()
        self._get()  # wait until the models have loaded

    def _get(self):
        while True:  # wait as long as the worker is alive; a crash raises instead of hanging
            try:
                ok, out = self.res.get(timeout=5)
                break
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f"OCR process died (exit code {self.proc.exitcode})")
        if not ok:
            raise RuntimeError(out)
        return out

    def lines(self, img_bgr):
        if not self.proc.is_alive():  # e.g. crashed on the previous image
            self._start()
        self.req.put(img_bgr)
        return self._get()

    def close(self):
        if self.proc.is_alive():
            self.req.put(None)
            self.proc.join(timeout=10)


def _ocr_worker(req, res, lang, det_max_side, device):
    # PyTorch puts its own NVIDIA DLL folder on PATH when it loads; a worker
    # started afterwards inherits that and would pick up PyTorch's cuDNN
    os.environ["PATH"] = os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep)
        if not re.search(r"[\\/]torch[\\/]lib", p, re.I))
    try:
        ocr = load_ocr(lang, det_max_side, device)
        res.put((True, None))
    except Exception as e:
        res.put((False, f"OCR failed to load: {type(e).__name__}: {e}"))
        return
    while (img := req.get()) is not None:
        try:
            res.put((True, ocr_lines(img, ocr)))
        except Exception as e:
            res.put((False, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------- LLM calls
def _json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


class OpenAIBackend:
    def __init__(self, model, vision_model=None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model, self.vision_model = model, vision_model or model

    def chat(self, prompt, image_bgr=None, json_mode=False, max_tokens=256):
        content = [{"type": "text", "text": prompt}]
        if image_bgr is not None:
            ok, png = cv2.imencode(".png", image_bgr)
            b64 = base64.b64encode(png.tobytes()).decode("utf-8")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self.client.chat.completions.create(
            model=self.vision_model if image_bgr is not None else self.model,
            temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}], **kw)
        return resp.choices[0].message.content


class QwenBackend:
    """Local Qwen2.5-VL; one model handles both the text and the image calls."""

    def __init__(self, hf_id, load_in_4bit=False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        kw = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        else:
            # bf16 on GPU; fp32 on CPU, where bf16 is slow on most consumer chips
            kw["dtype"] = "auto" if torch.cuda.is_available() else torch.float32
        print(f"loading {hf_id} (cuda={torch.cuda.is_available()}, 4bit={load_in_4bit})")
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModelForImageTextToText.from_pretrained(hf_id, **kw).eval()

    def chat(self, prompt, image_bgr=None, json_mode=False, max_tokens=256):
        content = [{"type": "text", "text": prompt}]
        images = None
        if image_bgr is not None:
            content.insert(0, {"type": "image"})
            images = [Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new, skip_special_tokens=True)[0]


def label_fields(lines, llm):
    """Ask the LLM to copy each field's text; returns {field: str}."""
    listing = "\n".join(l["text"] for l in lines)
    ans = _json(llm.chat(LABEL_PROMPT + listing, json_mode=True))
    out = {}
    for k in ("name", "date", "degree"):
        v = ans.get(k) or ""
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        out[k] = str(v).strip()
    return out


def reread_crop(crop_bgr, llm):
    # Text reads well at ~64-160 px tall. Small crops are upscaled; large ones are
    # shrunk, because the vision model's memory grows with pixel count (a name crop
    # from a 5712x4284 scan asked for 14 GB on a 6 GB GPU)
    h, w = crop_bgr.shape[:2]
    s = min(64 / h if h < 64 else 1.0, 160 / h, 1600 / w)
    if s != 1.0:
        interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        crop_bgr = cv2.resize(crop_bgr, None, fx=s, fy=s, interpolation=interp)
    return _json(llm.chat(REREAD_PROMPT, image_bgr=crop_bgr, max_tokens=64))


# ---------------------------------------------------------------- matching
def _words(s):
    """Words worth matching on: 3+ letters, or any number."""
    return {w for w in re.findall(r"[a-z]+|\d+", s.lower()) if len(w) >= 3 or w.isdigit()}


def _squash(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _similar(a, b, thresh=0.75):
    return a == b or (min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= thresh)


def _hit(word, text):
    """Word appears in text: as a substring with spaces removed (OCR that ran
    words together, "DAYOFJUNE"), or as a near match ("KRANA" ~ "ERANA")."""
    return word in _squash(text) or any(_similar(word, w) for w in _words(text))


def locate(value, lines):
    """Ids of the OCR line(s) an LLM answer was copied from."""
    want = _words(value)
    if not want:
        return []

    def gain(l, have):
        return sum(1 for w in want - have if _hit(w, l["text"]))

    # Most matching words wins; on a tie prefer the larger text, so the printed
    # name beats a repeat of it in micro-print at the bottom of the certificate
    seed = max(lines, key=lambda l: (gain(l, set()), l["box"][3] - l["box"][1]))
    if gain(seed, set()) == 0 and \
            SequenceMatcher(None, _squash(value), _squash(seed["text"])).ratio() < 0.5:
        return []
    ids = [seed["id"]]
    have = {w for w in want if _hit(w, seed["text"])}
    # A field can wrap onto neighbouring lines: extend while they add missing words
    for step in (-1, 1):
        j = seed["id"] + step
        while 0 <= j < len(lines) and want - have and gain(lines[j], have) >= min(2, len(want - have)):
            have |= {w for w in want if _hit(w, lines[j]["text"])}
            ids.append(j)
            j += step
    return sorted(ids)


def ocr_span(value, line_text):
    """The run of words in the OCR text that the LLM answer covers, verbatim.

    The LLM's copy is not reliable as a record of what is printed (it changes
    case, adds spaces, drops words), so the field text is taken from OCR.
    """
    toks = line_text.split()
    want = _words(value)
    hits = [i for i, t in enumerate(toks) if any(_hit(w, t) for w in want)]
    if not hits:
        return line_text
    return " ".join(toks[hits[0]:hits[-1] + 1]).strip(" ,.;:")


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def text_box(value, sel, fallback):
    """Tight box around just the field's words, e.g. only the name in
    "ERANA VEERAPPA DINESH S has been admitted to the DEGREE OF"."""
    words = [w for l in sel for w in l.get("words", [])]
    want, short = _words(value), {_squash(t) for t in value.split()}
    strong = [i for i, w in enumerate(words) if any(_hit(v, w["text"]) for v in want)]
    if not strong:
        return fallback
    lo, hi = strong[0], strong[-1]
    # Pull in short words the matching skips ("S" in "DINESH S", "of")
    while lo > 0 and _squash(words[lo - 1]["text"]) in short:
        lo -= 1
    while hi < len(words) - 1 and _squash(words[hi + 1]["text"]) in short:
        hi += 1
    return _union([w["box"] for w in words[lo:hi + 1]])


def field_from(value, lines):
    ids = locate(value, lines)
    if not ids:
        return None
    sel = [lines[i] for i in ids]
    box = _union([l["box"] for l in sel])
    line_text = " ".join(l["text"] for l in sel)
    text = ocr_span(value, line_text)
    return {"text": text, "llm_text": value, "box": box, "text_box": text_box(value, sel, box),
            "line_ids": ids, "line_text": line_text,
            "partial_line": len(_squash(text)) < 0.8 * len(_squash(line_text)),
            "min_conf": min(l["conf"] for l in sel)}


def pad_crop(img_bgr, box):
    x0, y0, x1, y1 = box
    py = int((y1 - y0) * 0.1)   # little vertical padding: the line above confuses the VLM
    px = y1 - y0                # about one character width of horizontal slack
    h, w = img_bgr.shape[:2]
    return img_bgr[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]


def _norm_name(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def name_ocr_ok(field, min_conf):
    return field["min_conf"] >= min_conf and bool(NAME_RE.match(field["text"].strip()))


# ---------------------------------------------------------------- dates
_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_IRREGULAR_ORD = {"first": "one", "second": "two", "third": "three", "fifth": "five",
                  "eighth": "eight", "ninth": "nine", "twelfth": "twelve"}


def _word_val(w):
    """Cardinal or ordinal number word -> int ("sixteenth" -> 16), else None."""
    w = _IRREGULAR_ORD.get(w, w)
    if w.endswith("ieth"):
        w = w[:-4] + "y"
    elif w.endswith("th") and w[:-2] in _UNITS:
        w = w[:-2]
    if w in _UNITS:
        return _UNITS.index(w)
    if w in _TENS[2:]:
        return _TENS.index(w) * 10
    return None


def _run_value(run):
    vals = [_word_val(w) for w in run if w not in ("and", "thousand", "hundred")]
    # "nineteen ninety eight" / "twenty twenty three": two-part year
    if "thousand" not in run and "hundred" not in run and len(vals) >= 2 \
            and vals[0] >= 10 and vals[1] >= 10:
        return vals[0] * 100 + sum(vals[1:])
    total = cur = 0
    for w in run:
        if w == "thousand":
            total, cur = total + max(cur, 1) * 1000, 0
        elif w == "hundred":
            cur = max(cur, 1) * 100
        elif w != "and":
            cur += _word_val(w)
    return total + cur


def words_to_digits(text):
    """'sixteenth day of August, two thousand and twenty-three' -> '16 day of august, 2023'."""
    words = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text.lower()).split()
    out, run = [], []

    def flush():
        tail = []
        while run and run[-1] == "and":
            tail.append(run.pop())
        if run:
            out.append(str(_run_value(run)))
        out.extend(tail)
        run.clear()

    for w in words:
        core = w.rstrip(",.;:")
        if _word_val(core) is not None or (run and core in ("thousand", "hundred", "and")):
            run.append(core)
            if core != w:           # punctuation ends the number
                flush()
                out[-1] += w[len(core):]
            continue
        flush()
        out.append(w)
    flush()
    return " ".join(out)


# "the 15th day of June, in the year of 2022" -> "15 June 2022"
FORMAL_DATE = re.compile(
    r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+day\s+of\s+([A-Za-z]+)[,\s]+(?:in\s+the\s+year\s+(?:of\s+)?)?(\d{4})",
    re.I)


def parse_date(text):
    """Returns (iso, precision) with precision 'day' or 'month', or (None, None)."""
    try:
        import dateparser
        from dateparser.search import search_dates
    except ImportError:
        return None, None
    text = words_to_digits(text)
    m = FORMAL_DATE.search(text)
    if m:
        text = " ".join(m.groups())
    for parts, precision, fmt in ((["day", "month", "year"], "day", "%Y-%m-%d"),
                                  (["month", "year"], "month", "%Y-%m")):
        settings = {"PREFER_DAY_OF_MONTH": "first", "REQUIRE_PARTS": parts}
        d = dateparser.parse(text, settings=settings)
        if d is None:
            # Date embedded in a sentence ("Given in Shanghai, ... June 15, 2022")
            found = search_dates(text, languages=["en"], settings=settings) or []
            d = found[0][1] if found else None
        if d:
            return d.strftime(fmt), precision
    return None, None


def degree_level(text):
    t = text.lower()
    squashed = re.sub(r"[^a-z]", "", t)
    for level, keywords in DEGREE_LEVELS:
        for kw in keywords:
            if (len(kw) >= 6 and kw in squashed) or re.search(rf"\b{re.escape(kw)}\b", t):
                return level
    return "unknown"


# ---------------------------------------------------------------- extract
def extract(img_bgr, ocr, llm, args):
    lines = ocr.lines(img_bgr)
    if not lines:
        return {"fields": {"name": None, "date": None, "degree": None}, "lines": []}
    answers = label_fields(lines, llm)
    fields = {k: field_from(answers[k], lines) if answers[k] else None
              for k in ("name", "date", "degree")}

    name = fields["name"]
    if name:
        name["ocr_text"] = name["text"]
        ocr_ok = name_ocr_ok(name, args.min_conf)
        reason = "always" if args.reread == "always" else (None if ocr_ok else "suspect")
        name["reread"] = reason
        if reason:
            vlm = reread_crop(pad_crop(img_bgr, name["box"]), llm)
            vtext = " ".join(str(vlm.get("text") or "").split())  # VLM sometimes adds newlines
            name["vlm_text"] = vtext or None
            name["vlm_legible"] = vlm.get("legible")
            # Take the vision read when OCR backs up every word of it (it fixes
            # misread letters and words the LLM dropped without inventing any),
            # or when OCR itself is unreliable
            supported = vtext and all(_hit(w, name["line_text"]) for w in _words(vtext))
            if vtext and vlm.get("legible", True) and (supported or not ocr_ok):
                name["text"] = vtext
            name["needs_review"] = not vtext or _norm_name(vtext) != _norm_name(name["ocr_text"])
        name["source"] = "vlm" if name["text"] != name["ocr_text"] else "ocr"
    date = fields["date"]
    if date:
        # The LLM's copy is often tidier ("June 15, 2022"), so try it first
        for t in (date["llm_text"], date["text"], date["line_text"]):
            date["iso"], date["precision"] = parse_date(t)
            if date["iso"]:
                break
    deg = fields["degree"]
    if deg:
        deg["level"] = degree_level(deg["llm_text"])
        if deg["level"] == "unknown":
            deg["level"] = degree_level(deg["text"])
    return {"fields": fields, "lines": lines}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Image file or folder of images")
    ap.add_argument("--output", default="fields.jsonl")
    ap.add_argument("--provider", choices=["qwen", "openai"], default="qwen")
    ap.add_argument("--model", help=f"qwen: HF id or local path (default {DEFAULT_QWEN}); "
                                    "openai: model id (required)")
    ap.add_argument("--vision_model", help="openai only: model id for the name re-read (default: --model)")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="qwen only: 4-bit quantisation so the 3B model fits a 6GB GPU")
    # Decorative fonts give confident misreads (blackletter "MARCEL" -> "HARCEL" at 0.89),
    # so by default every name gets a second read; "auto" only re-reads suspicious ones
    ap.add_argument("--reread", choices=["always", "auto"], default="always",
                    help="Vision re-read of the name: always, or only when OCR looks unreliable")
    ap.add_argument("--min_conf", type=float, default=0.95,
                    help="OCR names below this confidence defer to the vision read")
    ap.add_argument("--det_max_side", type=int, default=1600,
                    help="Longest side for OCR text detection; lower is faster, higher finds smaller text")
    ap.add_argument("--ocr_lang", help="PaddleOCR lang, e.g. 'en'; default is PaddleOCR's multilingual model")
    ap.add_argument("--ocr_device", choices=["auto", "gpu", "cpu"], default="auto",
                    help="auto: GPU if the GPU build of Paddle (paddlepaddle-gpu) is installed")
    ap.add_argument("--keep_lines", action="store_true", help="Also write all OCR lines")
    ap.add_argument("--resume", action="store_true",
                    help="Append to --output and skip images already in it (after a crash or stop)")
    args = ap.parse_args()

    if args.provider == "openai" and not args.model:
        ap.error("--model is required with --provider openai")
    # Start the OCR process before PyTorch loads (see OCRWorker)
    ocr = OCRWorker(args.ocr_lang, args.det_max_side, args.ocr_device)
    if args.provider == "openai":
        llm = OpenAIBackend(args.model, args.vision_model)
    else:
        llm = QwenBackend(args.model or DEFAULT_QWEN, args.load_in_4bit)

    src = Path(args.input)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in exts)

    done = set()
    if args.resume and Path(args.output).exists():
        for line in open(args.output, encoding="utf-8"):
            try:
                done.add(json.loads(line)["file"])
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"resume: {len(done)} images already in {args.output}, skipping them")

    failed = []
    with open(args.output, "a" if args.resume else "w", encoding="utf-8") as f:
        for path in files:
            if path.name in done:
                continue
            img = cv2.imread(str(path))
            if img is None:
                print(f"skip (unreadable): {path.name}")
                continue
            t0 = time.time()
            try:
                out = extract(img, ocr, llm, args)
            except Exception as e:  # e.g. GPU out of memory: log it and keep going
                print(f"FAIL: {path.name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
                failed.append(path.name)
                _free_gpu()
                continue
            rec = {"file": path.name, **out["fields"]}
            if args.keep_lines:
                rec["lines"] = out["lines"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n = out["fields"]["name"]
            flag = " [review]" if n and n.get("needs_review") else ""
            print(f"done: {path.name} -> {n['text'] if n else None!r}{flag} ({time.time() - t0:.0f}s)")
    ocr.close()
    if failed:
        print(f"\n{len(failed)} failed (not written; re-run with --resume to retry them): "
              + ", ".join(failed))


def _free_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
