"""
Generate a PDF showing sample images from results/fid_images.zip.

Layout:
  Section 1: Standard conditions — for each schedule, show actual vs virtual side-by-side,
              with N_SAMPLE images per condition in a grid.
  Section 2: Label-swap conditions — for each schedule, show virtual (no swap) vs
              swap_at_step1/3/5/7, with N_SAMPLE images per condition.

Output: results/fid_images.pdf
"""

import io
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ZIP_PATH = Path("results/fid_images.zip")
OUT_PDF   = Path("results/fid_images.pdf")

# How many sample images to show per condition row
N_SAMPLE = 8

# Schedules and their display names / size annotations
SCHEDULES = {
    "full":          "full (512×10)",
    "early_small":   "early_small (256×5 → 512×5)",
    "gradual_steep": "gradual_steep (128→…→512)",
}

SWAP_STEPS = [1, 3, 5, 7]
SWAP_STEP_LABELS = {1: "swap @ step 2", 3: "swap @ step 4", 5: "swap @ step 6", 7: "swap @ step 8"}


def load_images(zf: zipfile.ZipFile, cond: str, n: int) -> list[Image.Image]:
    files = sorted(name for name in zf.namelist() if name.startswith(cond + "/") and name.endswith(".png"))
    imgs = []
    for fname in files[:n]:
        data = zf.read(fname)
        imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
    return imgs


def draw_image_row(ax_row, images: list[Image.Image]) -> None:
    for ax, img in zip(ax_row, images):
        ax.imshow(img)
        ax.axis("off")
    # Hide unused axes
    for ax in ax_row[len(images):]:
        ax.set_visible(False)


def section_title_page(pdf: PdfPages, title: str) -> None:
    fig = plt.figure(figsize=(11, 2))
    fig.text(0.5, 0.5, title, ha="center", va="center", fontsize=20, fontweight="bold")
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_standard_page(pdf: PdfPages, zf: zipfile.ZipFile, sched_key: str, sched_label: str) -> None:
    """One page per schedule: actual (top half) vs virtual (bottom half)."""
    modes = [("actual", "Actual resize"), ("virtual", "Virtual resize")]
    n_rows = len(modes)
    fig, axes = plt.subplots(n_rows, N_SAMPLE, figsize=(N_SAMPLE * 1.5, n_rows * 1.7))
    fig.suptitle(f"Schedule: {sched_label}", fontsize=13, fontweight="bold", y=1.01)

    for row_idx, (mode, mode_label) in enumerate(modes):
        cond = f"{sched_key}_{mode}"
        imgs = load_images(zf, cond, N_SAMPLE)
        row_axes = axes[row_idx] if n_rows > 1 else axes
        # Row label on the leftmost axis
        row_axes[0].set_ylabel(mode_label, fontsize=9, rotation=90, labelpad=4)
        draw_image_row(row_axes, imgs)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_swap_page(pdf: PdfPages, zf: zipfile.ZipFile, sched_key: str, sched_label: str) -> None:
    """One page per schedule: no-swap virtual + 4 swap conditions."""
    row_defs = [("virtual", "No swap (virtual)")] + [
        (f"virtual_swap_at_step{s}", SWAP_STEP_LABELS[s]) for s in SWAP_STEPS
    ]
    n_rows = len(row_defs)
    fig, axes = plt.subplots(n_rows, N_SAMPLE, figsize=(N_SAMPLE * 1.5, n_rows * 1.7))
    fig.suptitle(f"Label-swap — Schedule: {sched_label}", fontsize=13, fontweight="bold", y=1.01)

    for row_idx, (mode_suffix, row_label) in enumerate(row_defs):
        cond = f"{sched_key}_{mode_suffix}"
        imgs = load_images(zf, cond, N_SAMPLE)
        row_axes = axes[row_idx] if n_rows > 1 else axes
        row_axes[0].set_ylabel(row_label, fontsize=8, rotation=90, labelpad=4)
        draw_image_row(row_axes, imgs)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf, PdfPages(OUT_PDF) as pdf:
        # ---- Section 1: standard conditions ----
        section_title_page(pdf, "Section 1: Actual vs Virtual Resize")
        for sched_key, sched_label in SCHEDULES.items():
            print(f"  standard: {sched_key}")
            make_standard_page(pdf, zf, sched_key, sched_label)

        # ---- Section 2: label-swap conditions ----
        section_title_page(pdf, "Section 2: Label-Swap (Virtual Resize)")
        for sched_key, sched_label in SCHEDULES.items():
            print(f"  label-swap: {sched_key}")
            make_swap_page(pdf, zf, sched_key, sched_label)

    print(f"PDF written to {OUT_PDF}")


if __name__ == "__main__":
    main()
