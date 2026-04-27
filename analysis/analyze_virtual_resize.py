"""
Analyze and plot results/virtual_resize_experiment.pkl.

Page 1 – three panels: latent_compress_output | latent_compress_targets | latent_no_compress
Page 2 – difference panel (compress_output − no_compress) + combined overlay panel
         (compress_output solid, no_compress dashed, same colors per image size)

Output: results/virtual_resize_analysis.pdf
"""

import pickle
import matplotlib
import matplotlib.lines
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PKL_PATH = "results/virtual_resize_experiment.pkl"
OUT_PDF  = "results/virtual_resize_analysis.pdf"

PANELS = [
    ("latent_compress_output",  "Compress output"),
    ("latent_compress_targets", "Compress targets"),
    ("latent_no_compress",      "No compress"),
]

# Perceptually distinct, print-friendly palette (tab10 subset, skipping red/green clash)
SIZE_CMAP = plt.cm.tab10


def load_data(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)["virtual_resize"]["dit"]


def mean_loss(dit: dict, metric: str) -> tuple[list[int], list[int], np.ndarray]:
    """Return (timesteps, latent_sizes, matrix[T, S]) of mean losses."""
    timesteps = sorted(dit[metric].keys())
    sizes = sorted(dit[metric][timesteps[0]].keys(), reverse=True)
    mat = np.array([
        [np.mean(dit[metric][t][s]) for s in sizes]
        for t in timesteps
    ])
    return timesteps, sizes, mat


def size_color(si: int, n: int):
    return SIZE_CMAP(si / max(n - 1, 1))


def add_legend(ax, sizes):
    n = len(sizes)
    handles = [
        matplotlib.lines.Line2D([0], [0], color=size_color(si, n), linewidth=2.0)
        for si, _ in enumerate(sizes)
    ]
    ax.legend(handles, [str(s * 8) for s in sizes],
              title="Image size", fontsize=8, title_fontsize=9,
              loc="upper right", framealpha=0.85)


def main():
    print(f"Loading {PKL_PATH} ...")
    dit = load_data(PKL_PATH)

    # ------------------------------------------------------------------ #
    # Page 1 – three individual panels
    # ------------------------------------------------------------------ #
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig1.suptitle(
        "Latent loss vs denoising timestep — lines per image size",
        fontsize=13,
    )

    sizes = None
    for ax, (metric, title) in zip(axes1, PANELS):
        timesteps, sizes, mat = mean_loss(dit, metric)
        n = len(sizes)
        for si, size in enumerate(sizes):
            ax.plot(timesteps, mat[:, si], color=size_color(si, n), linewidth=2.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_ylabel("Mean loss", fontsize=10)
        ax.set_xlim(timesteps[0], timesteps[-1])
        ax.set_ylim(0, mat.max() * 1.05)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)

    add_legend(axes1[-1], sizes)
    fig1.tight_layout()

    # ------------------------------------------------------------------ #
    # Page 2 – difference plot + combined overlay
    # ------------------------------------------------------------------ #
    timesteps, sizes, mat_comp   = mean_loss(dit, "latent_compress_output")
    _,         _,     mat_nocomp = mean_loss(dit, "latent_no_compress")
    diff = mat_comp - mat_nocomp
    n = len(sizes)

    fig2, (ax_diff, ax_comb) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(
        "Compress output vs no compress",
        fontsize=13,
    )

    # Left: difference
    ax_diff.set_title("Δ loss  (compress output − no compress)", fontsize=11)
    for si, size in enumerate(sizes):
        ax_diff.plot(timesteps, diff[:, si], color=size_color(si, n), linewidth=2.0)
    ax_diff.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_xlabel("Timestep", fontsize=10)
    ax_diff.set_ylabel("Δ loss", fontsize=10)
    ax_diff.set_xlim(timesteps[0], timesteps[-1])
    ax_diff.grid(True, alpha=0.25)
    ax_diff.tick_params(labelsize=9)

    # Right: combined overlay (compress_output solid, no_compress dashed)
    ax_comb.set_title("Compress output (solid) vs no compress (dashed)", fontsize=11)
    for si, size in enumerate(sizes):
        color = size_color(si, n)
        ax_comb.plot(timesteps, mat_comp[:, si],   color=color, linewidth=2.0, linestyle="-")
        ax_comb.plot(timesteps, mat_nocomp[:, si], color=color, linewidth=2.0, linestyle="--")
    ax_comb.set_xlabel("Timestep", fontsize=10)
    ax_comb.set_ylabel("Mean loss", fontsize=10)
    ax_comb.set_xlim(timesteps[0], timesteps[-1])
    ax_comb.set_ylim(0, max(mat_comp.max(), mat_nocomp.max()) * 1.05)
    ax_comb.grid(True, alpha=0.25)
    ax_comb.tick_params(labelsize=9)

    add_legend(ax_comb, sizes)
    fig2.tight_layout()

    with PdfPages(OUT_PDF) as pdf:
        pdf.savefig(fig1, bbox_inches="tight")
        pdf.savefig(fig2, bbox_inches="tight")
    plt.close("all")

    print(f"Done. PDF written to {OUT_PDF}")


if __name__ == "__main__":
    main()
