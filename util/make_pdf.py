"""
Arrange FID images from fid_images.zip into a PDF.

Layout: one row per image index, 10 columns.
  Left 5  columns: actual resize  (full, early_small, gradual_mild, gradual_medium, gradual_steep)
  Right 5 columns: virtual resize (same order)

Images are placed at full resolution (1px = 1pt); the page width expands to fit.
"""

import io
import zipfile

from PIL import Image
from reportlab.pdfgen import canvas

ZIP_PATH = "results/fid_images.zip"
OUT_PDF = "results/fid_images.pdf"

SCHEDULE_ORDER = ["full", "early_small", "gradual_mild", "gradual_medium", "gradual_steep"]
MODES = ["actual", "virtual"]  # left 5, right 5

GAP = 4  # px gap between images and between columns
HEADER_PT = 28  # points reserved for column labels at the top of each page


def load_images(zip_path: str) -> dict[str, dict[str, Image.Image]]:
    """Returns {cond_name: {filename: PIL.Image}}."""
    result: dict[str, dict[str, Image.Image]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".png"):
                continue
            parts = name.split("/")
            cond, fname = parts[0], parts[1]
            img = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
            result.setdefault(cond, {})[fname] = img
    return result


def main() -> None:
    data = load_images(ZIP_PATH)

    # Column order: actual × 5, then virtual × 5
    col_names = [f"{s}_{m}" for m in MODES for s in SCHEDULE_ORDER]

    # Gather sorted filenames (same across all conditions)
    sample_cond = col_names[0]
    fnames = sorted(data[sample_cond].keys())
    n_images = len(fnames)
    n_cols = len(col_names)  # 10

    # Image dimensions (assume uniform)
    sample_img = next(iter(data[sample_cond].values()))
    img_w, img_h = sample_img.size  # pixels = points at 1px/pt

    page_w = n_cols * img_w + (n_cols - 1) * GAP
    row_h = img_h + GAP
    page_h = n_images * row_h + HEADER_PT

    c = canvas.Canvas(OUT_PDF, pagesize=(page_w, page_h))
    c.setTitle("FID Images: actual (left) vs virtual (right) resize")

    # Column header labels (short names, drawn once at top)
    short_labels = [f"{s}\n{m}" for m in MODES for s in SCHEDULE_ORDER]
    c.setFont("Helvetica-Bold", 7)
    for col_idx, label in enumerate(short_labels):
        x = col_idx * (img_w + GAP)
        # reportlab y=0 is bottom; top of page = page_h
        y_text = page_h - HEADER_PT + 4
        for line_idx, line in enumerate(label.split("\n")):
            c.drawString(x + 2, y_text - line_idx * 10, line)

    # Draw separator between actual and virtual halves
    sep_x = (n_cols // 2) * (img_w + GAP) - GAP // 2
    c.setStrokeColorRGB(0.3, 0.3, 0.8)
    c.setLineWidth(2)
    c.line(sep_x, 0, sep_x, page_h)

    for row_idx, fname in enumerate(fnames):
        # y origin for this row (reportlab y grows upward)
        y = page_h - HEADER_PT - (row_idx + 1) * row_h + GAP

        for col_idx, cond in enumerate(col_names):
            img = data[cond][fname]
            x = col_idx * (img_w + GAP)

            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG", quality=95)
            img_bytes.seek(0)
            pil_img = Image.open(img_bytes)

            # Draw via reportlab image
            img_reader = io.BytesIO()
            img.save(img_reader, format="PNG")
            img_reader.seek(0)
            c.drawImage(
                __import__("reportlab.lib.utils", fromlist=["ImageReader"]).ImageReader(img_reader),
                x, y, width=img_w, height=img_h,
            )

    c.save()
    print(f"PDF written to {OUT_PDF}  ({n_images} rows × {n_cols} columns, {page_w}×{page_h:.0f} pt)")


if __name__ == "__main__":
    main()
