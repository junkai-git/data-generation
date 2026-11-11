# data_generation

Builds tampered certificate images for training forgery detectors (TruFor, ASCFormer).

1. **Extract** the recipient name, award date and degree from each clean certificate, along with where they are on the image.
2. **Edit** the name with an AI text-editing model (AnyText) so it reads as a different person, in the original lettering style.
3. **Verify** each edit with OCR and keep a pixel-accurate mask of what changed.

```
clean certificates
      │
      ▼
extract_fields.py ──► fields.jsonl          (.venv: PaddleOCR + Qwen2.5-VL)
      │
      ▼
edit_names.py generate ──► tampered/        (AnyText env, GPU with 8 GB+)
      │                    images/ masks/ metadata.jsonl
      ▼
edit_names.py verify ──► metadata_verified.jsonl   (.venv)
```

> **Status:** extraction and the `--dry_run` placement check have been tested on Windows. `edit_names.py generate` (the AnyText step) has **not been run yet**: the laptop's 6 GB GPU is too small. Expect to fix small issues on its first run.

---

## Folder contents

| File | Purpose | Runs in |
|---|---|---|
| `extract_fields.py` | OCR + Qwen: finds name / date / degree, their text and boxes → `fields.jsonl` | `.venv` |
| `edit_names.py` | `generate`: swaps names with AnyText. `verify`: OCR-checks the edits | AnyText env / `.venv` |
| `fields_to_excel.py` | `fields.jsonl` → readable Excel workbook, with review flags highlighted | `.venv` |
| `visualize_boxes.py` | Draws the field boxes on the certificates, saves box masks | `.venv` |
| `requirements.txt` | Packages for `.venv` | |
| `dataset/` | Certificates (`Dataset Certs/Negative Samples` = clean, `Positive Samples`) | |

---

## What you need

| Step | Hardware | Software |
|---|---|---|
| Extract + verify | NVIDIA GPU with **6 GB+** recommended (CPU works, much slower) | Python **3.10** |
| AnyText editing | NVIDIA GPU with **8 GB+**, 12 GB+ recommended. **Linux** is easiest | Miniconda |

One machine with a 12 GB+ GPU can do everything. Otherwise run extraction on one machine and AnyText on another, and copy the files between them.

---

## Part 1: Set up `.venv` (extraction + verify)

Use **Python 3.10**. `paddlepaddle` has no packages for newer versions such as 3.14.

Windows (PowerShell):
```powershell
cd path\to\data_generation
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux:
```bash
cd path/to/data_generation
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the GPU build of PyTorch for CUDA 12.8 (`+cu128`, from PyTorch's own index). If the new machine needs a different CUDA version, change the `--extra-index-url` line and the `torch` / `torchvision` versions to match. Use the selector on pytorch.org to find them.

Check the install:
```bash
python -c "import transformers, paddleocr, dateparser, accelerate, bitsandbytes; print('ok')"
python -c "import torch; print(torch.cuda.is_available())"      # should print True
```

---

## Part 2: Extract the fields

```bash
python extract_fields.py --input "dataset/Dataset Certs/Negative Samples" --output fields.jsonl --load_in_4bit
```

- The first run downloads Qwen2.5-VL-3B (about 7.5 GB, into the Hugging Face cache) and the PaddleOCR models.
- The startup line must say `cuda=True`. If it says `cuda=False`, see Troubleshooting.
- Expect roughly 45–80 s per certificate. Almost all of it is PaddleOCR on the CPU.
- Results are written as each image finishes. Each run **overwrites** `--output`.
- `--input` takes one image or a folder. Subfolders are **not** searched.

Useful options:

| Option | Default | What it does |
|---|---|---|
| `--load_in_4bit` | off | 4-bit Qwen; needed to fit a 6 GB GPU |
| `--provider openai --model gpt-4o-mini` | qwen | Use the OpenAI API instead (needs `OPENAI_API_KEY`, paid) |
| `--reread auto` | `always` | Only re-read names that look unreliable (faster) |
| `--keep_lines` | off | Also save every OCR line (for debugging) |

Each line of `fields.jsonl` is one certificate:
- `name`, `date`, `degree` each have:
  - `text`: as printed, read by OCR
  - `llm_text`: Qwen's copy
  - `box`: the whole OCR line
  - `text_box`: just the field's words
  - `min_conf`: OCR confidence
- `name` also has `ocr_text`, `vlm_text` (Qwen's image re-read), `source` and `needs_review` (the two readings disagree).
- `date` has `iso` (`YYYY-MM-DD`, or `YYYY-MM`) and `precision`.
- `degree` has `level` (bachelor, master, doctorate, …).

Check the results:
```bash
python fields_to_excel.py --input fields.jsonl
python visualize_boxes.py --jsonl fields.jsonl --images "dataset/Dataset Certs/Negative Samples"
```

---

## Part 3: Set up AnyText (on the GPU machine)

1. Install **Miniconda** if the machine doesn't have conda.
2. Clone AnyText (https://github.com/tyxsspa/AnyText.git) **next to** this folder, not inside it:
   ```bash
   git clone https://github.com/tyxsspa/AnyText.git
   ```
3. Add a font. AnyText expects `AnyText/font/Arial_Unicode.ttf`. For English names, regular Arial is enough. Copy it in under that name:
   - Windows: `Copy-Item C:\Windows\Fonts\arial.ttf AnyText\font\Arial_Unicode.ttf`
   - Linux: copy any Latin TTF, e.g. `cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf AnyText/font/Arial_Unicode.ttf`

   Use a broader font such as Noto Sans if names have accents or non-Latin letters.
4. Create the environment:
   ```bash
   cd AnyText
   conda env create -f environment.yaml
   conda activate anytext
   pip install faker
   ```
5. Test AnyText on its own. The first run downloads about 6 GB of model files to `~/.cache/modelscope/hub`; set `MODELSCOPE_CACHE` to put them elsewhere.
   ```bash
   python inference.py
   ```
   If that writes images, the environment works.

Optional, once it works: delete `demo.py`, `train.py`, `tool_add_anytext.py`, `lora_util.py`, `style.css`, `eval/`, `javascript/`, `docs/` and `example_images/` from the AnyText folder. Keep everything else.

---

## Part 4: Generate the edited certificates (AnyText env)

Copy to the GPU machine: `edit_names.py`, `fields.jsonl` and the certificate images. No other scripts from this folder are needed for this step.

**First check placement**, without AnyText. This writes each crop with the red edit region to `tampered/debug/`:
```bash
python edit_names.py generate --fields fields.jsonl --images "dataset/Dataset Certs/Negative Samples" --output_dir tampered --dry_run
```

**Then try a few certificates.** Make a small `fields_test.jsonl` with the first 3 lines of `fields.jsonl`:
```bash
python edit_names.py generate --fields fields_test.jsonl --images "dataset/Dataset Certs/Negative Samples" --output_dir tampered_test --anytext_dir ../AnyText
```

**Then the full run:**
```bash
python edit_names.py generate --fields fields.jsonl --images "dataset/Dataset Certs/Negative Samples" --output_dir tampered --anytext_dir ../AnyText --n_per_image 2
```

What it does per certificate:
1. Snaps the name's `text_box` to the real gaps between words.
2. Crops around it (AnyText works at ≤ 768 px).
3. Picks a fake name of similar length and the same case, from Faker or `--names_file`. Names are at most 20 characters (AnyText's limit).
4. AnyText writes the new name in the box.
5. Pastes back **only** the name region.
6. Mask = pixels that actually changed.

| Option | Default | What it does |
|---|---|---|
| `--n_per_image` | 1 | Fakes per certificate |
| `--names_file` | Faker | One name per line |
| `--steps` / `--cfg_scale` | 20 / 9.0 | AnyText sampling settings |
| `--fp16` | off | Half precision, if your AnyText version supports it |
| `--diff_thresh` | 20 | Pixel change (0–255) counted as tampered in the mask |
| `--seed` | 0 | Makes runs repeatable |

Output:
```
tampered/
  images/C_001_clean_ai00.png
  masks/C_001_clean_ai00_mask.png     white = changed pixels
  metadata.jsonl                      old/new name, boxes, prompt, seed, ...
```
`metadata.jsonl` is **appended** to on every run, so delete it, or use a new `--output_dir`, when starting over.

Certificates are skipped (with a message) when no name was found, or when the name shares a line with other words and has no `text_box`. Re-run `extract_fields.py` to get `text_box` in that case.

---

## Part 5: Verify the edits (`.venv`)

Copy `tampered/` back if needed, then:
```bash
python edit_names.py verify --output_dir tampered
```
This OCRs each edited name and writes `tampered/metadata_verified.jsonl`. Each line has `verified`: `true` when the new name is readable and the old name is gone. Leave out `verified: false` images when training.

---
