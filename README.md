# data_generation

Builds tampered certificate images for training forgery detectors (TruFor, ASCFormer).

1. **Extract** the recipient name, award date and degree from each clean certificate, along with where they are on the image.
2. **Edit** the name with a locally hosted AI image editor (Qwen-Image-Edit-2511). It sees the original name and copies its font, size, colour and ink texture, so the new name looks like it was always there.
3. **Verify** each edit with OCR and keep a pixel-accurate mask of what changed.
4. **Train** once forged certificates are generated and validated, train on models provided by SMU (https://github.com/LifuWang-66/ControlDoc.git)

Everything runs locally, in one Python environment. No web services are used. Models are downloaded once from Hugging Face.

```
clean certificates
      │
      ▼
extract_fields.py ──► fields.jsonl          (PaddleOCR + Qwen2.5-VL, GPU 6 GB+)
      │
      ▼
edit_names.py generate ──► tampered/        (Qwen-Image-Edit-2511, GPU 24 GB+)
      │                    images/ masks/ metadata.jsonl
      ▼
edit_names.py verify ──► metadata_verified.jsonl   (PaddleOCR)
```

> **Status:** extraction, the `--dry_run` placement check, and the shift-correction and masking steps have been tested on Windows. The actual Qwen-Image-Edit run has **not been tested yet**: the laptop's 6 GB GPU is far too small. Expect to fix small issues on its first run.

---

## Folder contents

| File | Purpose | Runs in |
|---|---|---|
| `extract_fields.py` | OCR + Qwen: finds name / date / degree, their text and boxes → `fields.jsonl` | `.venv` |
| `edit_names.py` | `generate`: swaps names with Qwen-Image-Edit. `verify`: OCR-checks the edits | `.venv` |
| `fields_to_excel.py` | `fields.jsonl` → readable Excel workbook, with review flags highlighted | `.venv` |
| `visualize_boxes.py` | Draws the field boxes on the certificates, saves box masks | `.venv` |
| `requirements.txt` | Packages for `.venv` | |
| `dataset/` | Certificates (`Dataset Certs/Negative Samples` = clean, `Positive Samples`) | |

---

## What you need

| Step | Hardware | Software |
|---|---|---|
| Extract + verify | NVIDIA GPU with **6 GB+** recommended (CPU works, much slower) | Python **3.10**, `git` |
| Name editing | NVIDIA GPU with **24 GB+** (see the memory table in Part 4), 64 GB+ system RAM, about 60 GB free disk | same `.venv` |

One machine with a 24 GB+ GPU can do everything. Otherwise run extraction on a smaller machine and editing on a bigger one, with the same `requirements.txt`, and copy the files between them.

---

## Part 1: Set up `.venv`

Use **Python 3.10**. `paddlepaddle` has no packages for newer versions such as 3.14. `git` must be installed, because `diffusers` is installed straight from GitHub.

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
python -c "import transformers, paddleocr, dateparser, accelerate, bitsandbytes, faker; from diffusers import QwenImageEditPlusPipeline; print('ok')"
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

<img width="1023" height="1125" alt="image" src="https://github.com/user-attachments/assets/a51055ba-33be-4f02-95b3-cdfffe770374" />


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

## Part 3: Check the names before editing

Open `fields.xlsx` (from `fields_to_excel.py`) and look at names highlighted yellow (`needs_review`). Some are fine, but some are not names at all. For example, on one certificate a **signature** was read as "CRAIG", and editing it would write a fake name over the signatures. Then either:
- fix or delete those lines in `fields.jsonl`, or
- pass `--skip_review` in Part 5 to skip every flagged name. That's simpler, but drops more certificates (21 of 54 in the first batch).

Also check placement without loading the editing model. This writes each crop with the red paste region to `tampered/debug/`:
```bash
python edit_names.py generate --fields fields.jsonl --images "dataset/Dataset Certs/Negative Samples" --output_dir tampered --dry_run
```

---

## Part 4: Get the image editor (Qwen-Image-Edit-2511)

The model is open (Apache 2.0) and runs locally through `diffusers`, which `requirements.txt` already installs. It's large: about 20B parameters, roughly **55–60 GB** on disk.

**Machine with internet:** nothing to do. The first `generate` run downloads it into the Hugging Face cache. To download it ahead of time:
```bash
hf download Qwen/Qwen-Image-Edit-2511
```

**Machine without internet:** download it on a machine that has internet, into a folder:
```bash
hf download Qwen/Qwen-Image-Edit-2511 --local-dir qwen-image-edit-2511
```
Copy that folder to the GPU machine. Then run Part 5 with `--model path/to/qwen-image-edit-2511`, after:
```bash
export HF_HUB_OFFLINE=1
```

**How much GPU memory you need.** These are estimates; the 4-bit route comes from community reports, not the official docs:

| GPU memory | Options to add |
|---|---|
| 80 GB (A100 / H100) | none |
| 48 GB | `--offload` |
| 24 GB | `--quantize 4bit` (add `--offload` if it runs out of memory) |
| 16 GB | `--quantize 4bit --offload` (may work, slow) |

`--offload` keeps parts of the model in system RAM until they're needed: slower, but much lighter on the GPU. `--quantize 4bit` shrinks the model with a small quality cost.

---

## Part 5: Generate the edited certificates

**Try a few certificates first.** Make a small `fields_test.jsonl` with the first 3 lines of `fields.jsonl`:
```bash
python edit_names.py generate --fields fields_test.jsonl --images "dataset/Dataset Certs/Negative Samples" --output_dir tampered_test
```
Open the results in `tampered_test/images` and compare them with the originals before running everything.

**Then the full run:**
```bash
python edit_names.py generate --fields fields.jsonl --images "dataset/Dataset Certs/Negative Samples" --output_dir tampered --n_per_image 2
```

What it does per certificate:
1. Snaps the name's `text_box` to the real gaps between words.
2. Crops around the name, with some surrounding certificate for context.
3. Picks a fake name of similar length and the same case, from Faker or `--names_file`.
4. Asks the model: *Change the name "OLD" to "NEW". Keep exactly the same font, letter style, size, weight, colour, spacing and position.* The model sees the original name, so it can copy its lettering.
5. Undoes any small shift the model introduced, then pastes back **only** the name region. The model slightly redraws everything it sees, so the rest of the certificate must stay untouched.
6. Mask = pixels that actually changed.

| Option | Default | What it does |
|---|---|---|
| `--n_per_image` | 1 | Fakes per certificate |
| `--names_file` | Faker | One name per line |
| `--skip_review` | off | Skip names flagged `needs_review` (see Part 3) |
| `--model` | `Qwen/Qwen-Image-Edit-2511` | Hugging Face id or local folder |
| `--quantize 4bit` / `--offload` | off | Use less GPU memory (see Part 4) |
| `--steps` / `--cfg_scale` | 40 / 4.0 | Sampling settings. More steps can be cleaner but slower |
| `--prompt` | see above | Your own instruction; must contain `{old}` and `{new}` |
| `--diff_thresh` | 20 | Pixel change (0–255) counted as tampered in the mask |
| `--max_spill` | 0.5 | Warn when the model changed a lot outside the name |
| `--seed` | 0 | Makes runs repeatable |

Output:
```
tampered/
  images/C_001_clean_ai00.png
  masks/C_001_clean_ai00_mask.png     white = changed pixels
  metadata.jsonl                      old/new name, boxes, prompt, seed, shift, spill_ratio, ...
```
`metadata.jsonl` is **appended** to on every run, so delete it, or use a new `--output_dir`, when starting over.

- **A `[check: model also changed outside the name]` warning** means the model redrew other parts of the crop too, possibly drawing the name in a different place, and only the name region was pasted back. Look at that image. `spill_ratio` in `metadata.jsonl` records how much changed outside the name.
- **Certificates are skipped (with a message)** when no name was found, when `--skip_review` applies, or when the name shares a line with other words and has no `text_box`. Re-run `extract_fields.py` to get `text_box`.
- **One failed edit doesn't stop the run.** For example, running out of GPU memory on a very large crop prints `fail:` and continues.

---

## Part 6: Verify the edits

```bash
python edit_names.py verify --output_dir tampered
```
This OCRs each edited name and writes `tampered/metadata_verified.jsonl`. Each line has `verified`: `true` when the new name is readable and the old name is gone. Leave out `verified: false` images when training.

---
