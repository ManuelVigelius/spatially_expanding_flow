"""
Analyze results from results/trajectory_divergence.zip.

Produces a multi-panel PDF: results/trajectory_divergence_analysis.pdf

Figures
-------
1. Divergence curves  — per-injection-step MSE over remaining denoising steps,
   one subplot grid per (compression_size × stride), coloured by injection step.

2. Final MSE heatmap  — final-step MSE as a function of injection_step × stride,
   one heatmap per compression size.

3. Injection-step effect — final MSE vs injection step, lines per stride,
   one plot per compression size.

4. Stride effect         — final MSE vs stride, lines per injection step,
   one plot per compression size.

5. Image gallery         — side-by-side comparison of baseline vs a few
   representative perturbed conditions (comp128/comp256, early/late injection).
"""

import io
import pickle
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# ---------------------------------------------------------------------------
ZIP_PATH = "results/trajectory_divergence.zip"
OUT_PDF  = "results/trajectory_divergence_analysis.pdf"

COMP_SIZES    = [128, 256]
INJ_STEPS     = list(range(4, 33, 4))   # [4, 8, ..., 32]
STRIDES       = [1, 2, 4]
N_STEPS       = 32

STRIDE_LABELS = {1: "stride 1 (every step)", 2: "stride 2 (every 2nd)", 4: "stride 4 (every 4th)"}
COMP_LABELS   = {128: "comp 128 (16×16 latent)", 256: "comp 256 (32×32 latent)"}

STEP_CMAP  = plt.cm.viridis
STRIDE_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}
# ---------------------------------------------------------------------------


def load_data(zip_path: str) -> tuple[dict, zipfile.ZipFile]:
    zf = zipfile.ZipFile(zip_path)
    results = pickle.loads(zf.read("results.pkl"))
    return results, zf


def step_norm(step: int) -> float:
    """Map injection step [4..32] → [0..1] for colormap."""
    return (step - INJ_STEPS[0]) / (INJ_STEPS[-1] - INJ_STEPS[0])


# ---------------------------------------------------------------------------
# Figure 1 – divergence curves
# ---------------------------------------------------------------------------

def fig_divergence_curves(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(
        len(STRIDES), len(COMP_SIZES),
        figsize=(14, 10),
        sharey=False,
    )
    fig.suptitle("Trajectory divergence over denoising steps\n(MSE vs baseline x₀, averaged over 80 images)", fontsize=13)

    for col, comp in enumerate(COMP_SIZES):
        for row, stride in enumerate(STRIDES):
            ax = axes[row][col]
            for inj in INJ_STEPS:
                entry = results[comp][inj][stride]
                xs = [s + 1 for s in entry["step_indices"]]  # 1-indexed steps
                ys = entry["divergence"]
                color = STEP_CMAP(step_norm(inj))
                ax.plot(xs, ys, color=color, linewidth=1.2, label=f"inj={inj}")
                # Mark injection point
                ax.axvline(inj, color=color, linewidth=0.5, linestyle=":", alpha=0.5)

            ax.set_title(f"{COMP_LABELS[comp]}\n{STRIDE_LABELS[stride]}", fontsize=8)
            ax.set_xlabel("Denoising step", fontsize=7)
            ax.set_ylabel("MSE", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_xlim(1, N_STEPS)
            ax.set_ylim(bottom=0)

    # Shared colorbar for injection step
    sm = plt.cm.ScalarMappable(cmap=STEP_CMAP, norm=plt.Normalize(vmin=INJ_STEPS[0], vmax=INJ_STEPS[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Injection step", fontsize=9)
    cbar.set_ticks(INJ_STEPS)

    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Figure 2 – final MSE heatmaps
# ---------------------------------------------------------------------------

def fig_heatmaps(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, len(COMP_SIZES), figsize=(12, 5))
    fig.suptitle("Final-step MSE (after full denoising from injection point)", fontsize=12)

    for ax, comp in zip(axes, COMP_SIZES):
        matrix = np.zeros((len(STRIDES), len(INJ_STEPS)))
        for ri, stride in enumerate(STRIDES):
            for ci, inj in enumerate(INJ_STEPS):
                matrix[ri, ci] = results[comp][inj][stride]["divergence"][-1]

        im = ax.imshow(matrix, aspect="auto", cmap="magma", origin="upper")
        ax.set_xticks(range(len(INJ_STEPS)))
        ax.set_xticklabels(INJ_STEPS, fontsize=8)
        ax.set_yticks(range(len(STRIDES)))
        ax.set_yticklabels([f"stride {s}" for s in STRIDES], fontsize=8)
        ax.set_xlabel("Injection step", fontsize=9)
        ax.set_title(COMP_LABELS[comp], fontsize=10)
        fig.colorbar(im, ax=ax, label="MSE")

        # Annotate cells
        for ri in range(len(STRIDES)):
            for ci in range(len(INJ_STEPS)):
                ax.text(ci, ri, f"{matrix[ri, ci]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if matrix[ri, ci] > matrix.max() * 0.5 else "black")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 – injection-step effect (final MSE vs injection step)
# ---------------------------------------------------------------------------

def fig_injection_effect(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, len(COMP_SIZES), figsize=(12, 4), sharey=False)
    fig.suptitle("Effect of injection timing on final-step MSE", fontsize=12)

    for ax, comp in zip(axes, COMP_SIZES):
        for stride in STRIDES:
            ys = [results[comp][inj][stride]["divergence"][-1] for inj in INJ_STEPS]
            ax.plot(INJ_STEPS, ys, marker="o", color=STRIDE_COLORS[stride],
                    label=f"stride {stride}", linewidth=1.5, markersize=5)

        ax.set_title(COMP_LABELS[comp], fontsize=10)
        ax.set_xlabel("Injection step", fontsize=9)
        ax.set_ylabel("Final-step MSE", fontsize=9)
        ax.set_xticks(INJ_STEPS)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 – stride effect (final MSE vs stride per injection step)
# ---------------------------------------------------------------------------

def fig_stride_effect(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, len(COMP_SIZES), figsize=(12, 4), sharey=False)
    fig.suptitle("Effect of stride on final-step MSE (per injection step)", fontsize=12)

    for ax, comp in zip(axes, COMP_SIZES):
        for inj in INJ_STEPS:
            ys = [results[comp][inj][stride]["divergence"][-1] for stride in STRIDES]
            color = STEP_CMAP(step_norm(inj))
            ax.plot(STRIDES, ys, marker="o", color=color, label=f"inj={inj}",
                    linewidth=1.2, markersize=4)

        ax.set_title(COMP_LABELS[comp], fontsize=10)
        ax.set_xlabel("Stride", fontsize=9)
        ax.set_ylabel("Final-step MSE", fontsize=9)
        ax.set_xticks(STRIDES)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    sm = plt.cm.ScalarMappable(cmap=STEP_CMAP, norm=plt.Normalize(vmin=INJ_STEPS[0], vmax=INJ_STEPS[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("Injection step", fontsize=9)
    cbar.set_ticks(INJ_STEPS)

    fig.tight_layout(rect=[0, 0, 0.93, 1.0])
    return fig


# ---------------------------------------------------------------------------
# Figure 5 – comp128 vs comp256: divergence curves for all injection steps,
#             one stride per row, two comp columns
# ---------------------------------------------------------------------------

def fig_comp_comparison(results: dict) -> plt.Figure:
    """Overlay comp128 vs comp256 on the same axes — all strides × all injection steps."""
    comp_colors = {128: "#d62728", 256: "#1f77b4"}

    fig, axes = plt.subplots(
        len(STRIDES), len(INJ_STEPS),
        figsize=(20, 8),
        sharey=False,
    )
    fig.suptitle("Comp 128 vs comp 256 — divergence curves (red = 128, blue = 256)", fontsize=12)

    for ri, stride in enumerate(STRIDES):
        for ci, inj in enumerate(INJ_STEPS):
            ax = axes[ri][ci]
            for comp in COMP_SIZES:
                entry = results[comp][inj][stride]
                xs = [s + 1 for s in entry["step_indices"]]
                ax.plot(xs, entry["divergence"], color=comp_colors[comp],
                        linewidth=1.2, label=f"comp {comp}")
            ax.axvline(inj, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(f"inj={inj}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlim(1, N_STEPS)
            ax.set_ylim(bottom=0)
            # Row label on leftmost column
            if ci == 0:
                ax.set_ylabel(f"stride {stride}\nMSE", fontsize=7)
            # Legend once
            if ri == 0 and ci == 0:
                ax.legend(fontsize=6)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6 – image gallery
# ---------------------------------------------------------------------------

def _load_img(zf: zipfile.ZipFile, arcname: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(zf.read(arcname))))


def figs_gallery(zf: zipfile.ZipFile) -> list[plt.Figure]:
    """
    Six pages of image galleries, one per (comp × stride) combination.
    Each page: rows = 9 (baseline + 8 injection steps), cols = 8 images of class 0000.
    """
    N_IMG = 8   # images per class (indices 00–07)
    CLASS = 0   # show class 0000 on every page

    col_labels = [f"img {i}" for i in range(N_IMG)]

    # First row is always the baseline
    baseline_folder = "baseline_images"
    baseline_label  = "Baseline\n(no injection)"

    figs = []
    for comp in COMP_SIZES:
        for stride in STRIDES:
            # rows: baseline + one per injection step
            row_folders = [baseline_folder] + [
                f"comp{comp}_step{inj:02d}_stride{stride}" for inj in INJ_STEPS
            ]
            row_labels = [baseline_label] + [
                f"inj = step {inj}" for inj in INJ_STEPS
            ]

            n_rows = len(row_folders)
            n_cols = N_IMG

            fig = plt.figure(figsize=(n_cols * 1.6 + 2.2, n_rows * 1.7 + 0.9))
            fig.suptitle(
                f"Image gallery — comp={comp} (latent {comp//8}×{comp//8}), stride={stride}\n"
                f"Rows: injection timing  |  Columns: 8 images of ImageNet class {CLASS:04d}",
                fontsize=10,
            )

            gs = gridspec.GridSpec(
                n_rows, n_cols,
                figure=fig,
                left=0.14, right=0.99,
                top=0.92, bottom=0.01,
                hspace=0.04, wspace=0.03,
            )
            axes = [[fig.add_subplot(gs[ri, ci]) for ci in range(n_cols)] for ri in range(n_rows)]

            for ri, folder in enumerate(row_folders):
                for ci in range(N_IMG):
                    ax = axes[ri][ci]
                    fname = f"class{CLASS:04d}_{ci:02d}.png"
                    arcname = f"trajectory_divergence/{folder}/{fname}"
                    try:
                        ax.imshow(_load_img(zf, arcname))
                    except KeyError:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax.transAxes, fontsize=7)
                    ax.axis("off")
                    if ri == 0:
                        ax.set_title(col_labels[ci], fontsize=7, pad=3)

            # Place row labels after draw so positions are finalised
            fig.canvas.draw()
            for ri, label in enumerate(row_labels):
                pos = axes[ri][0].get_position()
                y_centre = (pos.y0 + pos.y1) / 2
                fig.text(
                    0.005, y_centre, label,
                    ha="left", va="center",
                    fontsize=7.5, multialignment="left", linespacing=1.3,
                )

            figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Figure 7 – summary statistics table
# ---------------------------------------------------------------------------

def fig_summary_table(results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    fig.suptitle("Summary: final-step MSE across all conditions", fontsize=12)

    headers = ["Comp", "Inj step", "Stride 1 MSE", "Stride 2 MSE", "Stride 4 MSE",
               "Best stride", "MSE reduction vs stride4"]
    rows = []
    for comp in COMP_SIZES:
        for inj in INJ_STEPS:
            mses = {s: results[comp][inj][s]["divergence"][-1] for s in STRIDES}
            best = min(mses, key=mses.__getitem__)
            reduction = (mses[4] - mses[best]) / mses[4] * 100
            rows.append([
                str(comp),
                str(inj),
                f"{mses[1]:.4f}",
                f"{mses[2]:.4f}",
                f"{mses[4]:.4f}",
                str(best),
                f"{reduction:.1f}%",
            ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {ZIP_PATH} ...")
    results, zf = load_data(ZIP_PATH)

    out_path = OUT_PDF
    print(f"Generating analysis → {out_path}")

    with PdfPages(out_path) as pdf:
        print("  Figure 1: divergence curves ...")
        pdf.savefig(fig_divergence_curves(results), bbox_inches="tight")
        plt.close("all")

        print("  Figure 2: final MSE heatmaps ...")
        pdf.savefig(fig_heatmaps(results), bbox_inches="tight")
        plt.close("all")

        print("  Figure 3: injection-step effect ...")
        pdf.savefig(fig_injection_effect(results), bbox_inches="tight")
        plt.close("all")

        print("  Figure 4: stride effect ...")
        pdf.savefig(fig_stride_effect(results), bbox_inches="tight")
        plt.close("all")

        print("  Figure 5: comp128 vs comp256 comparison ...")
        pdf.savefig(fig_comp_comparison(results), bbox_inches="tight")
        plt.close("all")

        print("  Figures 6–11: image galleries (6 pages) ...")
        for i, fig in enumerate(figs_gallery(zf), start=6):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print("  Figure 7: summary table ...")
        pdf.savefig(fig_summary_table(results), bbox_inches="tight")
        plt.close("all")

    zf.close()
    print(f"\nDone. PDF written to {out_path}")

    # Also print a quick console summary
    print("\n=== Quick summary ===")
    print(f"{'Comp':>6}  {'Inj':>4}  {'s1 MSE':>8}  {'s2 MSE':>8}  {'s4 MSE':>8}")
    for comp in COMP_SIZES:
        for inj in INJ_STEPS:
            mses = [f"{results[comp][inj][s]['divergence'][-1]:.4f}" for s in STRIDES]
            print(f"{comp:>6}  {inj:>4}  {'  '.join(mses)}")


if __name__ == "__main__":
    main()
