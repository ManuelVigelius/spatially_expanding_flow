"""
Compute FID scores for generated images using clean-fid.

Compares images in results/fid_images/{full,optimal}/ against each other
and against the ImageNet validation set reference statistics.

Run after generate_fid_images.py has completed.
"""

import json
import os

import torch
from cleanfid import fid

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
OUT_DIR = "results/fid_images"
FULL_DIR = os.path.join(OUT_DIR, "full")
OPTIMAL_DIR = os.path.join(OUT_DIR, "optimal")

DATASET_NAME = "imagenet_train"   # clean-fid built-in reference stats
DATASET_RES = 256                 # closest available to DiT's 512px output
DATASET_SPLIT = "train"

BATCH_SIZE = 64
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    results = {}

    print("=" * 60)
    print(f"FID: full-resolution vs ImageNet reference")
    print("=" * 60)
    score_full = fid.compute_fid(
        fdir1=FULL_DIR,
        dataset_name=DATASET_NAME,
        dataset_res=DATASET_RES,
        dataset_split=DATASET_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )
    results["full_vs_imagenet"] = score_full
    print(f"FID (full vs ImageNet): {score_full:.4f}\n")

    print("=" * 60)
    print(f"FID: optimal-schedule vs ImageNet reference")
    print("=" * 60)
    score_optimal = fid.compute_fid(
        fdir1=OPTIMAL_DIR,
        dataset_name=DATASET_NAME,
        dataset_res=DATASET_RES,
        dataset_split=DATASET_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )
    results["optimal_vs_imagenet"] = score_optimal
    print(f"FID (optimal vs ImageNet): {score_optimal:.4f}\n")

    print("=" * 60)
    print("FID: full-resolution vs optimal-schedule")
    print("=" * 60)
    score_cross = fid.compute_fid(
        fdir1=FULL_DIR,
        fdir2=OPTIMAL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )
    results["full_vs_optimal"] = score_cross
    print(f"FID (full vs optimal): {score_cross:.4f}\n")

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  full    vs ImageNet : {results['full_vs_imagenet']:.4f}")
    print(f"  optimal vs ImageNet : {results['optimal_vs_imagenet']:.4f}")
    print(f"  full    vs optimal  : {results['full_vs_optimal']:.4f}")
    delta = results["optimal_vs_imagenet"] - results["full_vs_imagenet"]
    print(f"  FID degradation (optimal - full): {delta:+.4f}")

    json_path = os.path.join(OUT_DIR, "fid_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
