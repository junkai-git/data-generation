"""
Convert the field-extraction metadata (fields.jsonl) into a readable Excel workbook.

Workbook layout:
    Overview      one row per file: the final text of every field plus the most
                  useful attributes (date iso, degree level, review flags, confidence)
    <field>       one sheet per field (name, date, degree, ...) with every attribute
    Errors        lines that could not be parsed (only if there were any)

Fields are discovered from the data, so new fields/attributes added to the jsonl
show up automatically.

Highlighting:
    yellow  needs_review is true
    orange  min_conf below --low_conf (default 0.90)
    red     field missing / null for that file

Usage:
    pip install openpyxl
    python fields_to_excel.py
    python fields_to_excel.py --input fields.jsonl --output fields.xlsx --low_conf 0.85
"""

import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Attributes pulled into the Overview sheet (when a field has them), in this order.
OVERVIEW_ATTRS = ["iso", "precision", "level", "source", "needs_review", "min_conf"]
# Preferred column order for per-field sheets; anything else follows in first-seen order.
FIELD_ATTR_ORDER = ["text", "llm_text", "ocr_text", "vlm_text", "line_text",
                    "iso", "precision", "level", "source", "needs_review",
                    "vlm_legible", "reread", "min_conf", "partial_line",
                    "line_ids", "box"]

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
REVIEW_FILL = PatternFill("solid", fgColor="FFF2CC")
LOWCONF_FILL = PatternFill("solid", fgColor="F8CBAD")
MISSING_FILL = PatternFill("solid", fgColor="FF9999")
WRAP = Alignment(wrap_text=True, vertical="top")


def load_records(path):
    records, errors = [], []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError("line is not a JSON object")
                records.append(rec)
            except Exception as e:
                errors.append((lineno, str(e), line.strip()))
    return records, errors


def discover(records):
    """Return (field names, {field: [attr names]}) in first-seen order."""
    fields, attrs = [], {}
    for rec in records:
        for k, v in rec.items():
            if k == "file" or not (isinstance(v, dict) or v is None):
                continue
            if k not in fields:
                fields.append(k)
                attrs[k] = []
            if isinstance(v, dict):
                for a in v:
                    if a not in attrs[k]:
                        attrs[k].append(a)
    for k in fields:
        attrs[k].sort(key=lambda a: (FIELD_ATTR_ORDER.index(a) if a in FIELD_ATTR_ORDER
                                     else len(FIELD_ATTR_ORDER)))
    return fields, attrs


def cell_value(v):
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def write_header(ws, headers):
    ws.append(headers)
    for c in ws[1]:
        c.fill = HEADER_FILL
        c.font = HEADER_FONT
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"


def autosize(ws, max_width=50):
    for col in ws.columns:
        width = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(width + 2, 8), max_width)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = WRAP


def flag_fill(field_val, low_conf):
    """Pick the highlight for a field dict (None -> missing)."""
    if field_val is None:
        return MISSING_FILL
    if field_val.get("needs_review") is True:
        return REVIEW_FILL
    conf = field_val.get("min_conf")
    if isinstance(conf, (int, float)) and conf < low_conf:
        return LOWCONF_FILL
    return None


def build_overview(wb, records, fields, attrs, low_conf):
    ws = wb.active
    ws.title = "Overview"
    cols = []  # (field, attr) pairs
    for f in fields:
        cols.append((f, "text"))
        cols += [(f, a) for a in OVERVIEW_ATTRS if a in attrs[f]]
    write_header(ws, ["file"] + [f if a == "text" else f"{f} {a}" for f, a in cols])

    for r, rec in enumerate(records, 2):
        ws.cell(r, 1, rec.get("file", ""))
        for c, (f, a) in enumerate(cols, 2):
            fv = rec.get(f)
            cell = ws.cell(r, c, "(missing)" if fv is None and a == "text"
                           else cell_value(fv.get(a)) if isinstance(fv, dict) else "")
            if a == "text":
                fill = flag_fill(fv, low_conf)
                if fill:
                    cell.fill = fill
            elif a == "needs_review" and cell.value is True:
                cell.fill = REVIEW_FILL
            elif a == "min_conf" and isinstance(cell.value, (int, float)) and cell.value < low_conf:
                cell.fill = LOWCONF_FILL
    autosize(ws)


def build_field_sheet(wb, records, field, field_attrs, low_conf):
    ws = wb.create_sheet(field[:31])
    write_header(ws, ["file"] + field_attrs)
    for r, rec in enumerate(records, 2):
        ws.cell(r, 1, rec.get("file", ""))
        fv = rec.get(field)
        if not isinstance(fv, dict):
            ws.cell(r, 2, "(missing)").fill = MISSING_FILL
            continue
        for c, a in enumerate(field_attrs, 2):
            ws.cell(r, c, cell_value(fv.get(a)))
        fill = flag_fill(fv, low_conf)
        if fill:
            ws.cell(r, 1).fill = fill
    autosize(ws)


def build_legend(wb, low_conf):
    ws = wb.create_sheet("Legend")
    ws.append(["Colour", "Meaning"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for fill, text in [(REVIEW_FILL, "needs_review is true"),
                       (LOWCONF_FILL, f"min_conf below {low_conf}"),
                       (MISSING_FILL, "field missing / null")]:
        ws.append(["", text])
        ws.cell(ws.max_row, 1).fill = fill
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 30


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(here / "fields.jsonl"))
    ap.add_argument("--output", default=None, help="default: <input>.xlsx")
    ap.add_argument("--low_conf", type=float, default=0.90, help="highlight min_conf below this")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".xlsx")

    records, errors = load_records(in_path)
    fields, attrs = discover(records)

    wb = Workbook()
    build_overview(wb, records, fields, attrs, args.low_conf)
    for f in fields:
        build_field_sheet(wb, records, f, attrs[f], args.low_conf)
    if errors:
        ws = wb.create_sheet("Errors")
        write_header(ws, ["line", "error", "content"])
        for e in errors:
            ws.append(list(e))
        autosize(ws, max_width=100)
    build_legend(wb, args.low_conf)

    wb.save(out_path)
    print(f"Wrote {len(records)} records ({len(fields)} fields) to {out_path}")
    if errors:
        print(f"WARNING: {len(errors)} unparseable line(s), see 'Errors' sheet")


if __name__ == "__main__":
    main()
