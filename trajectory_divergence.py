"""
Trajectory divergence experiment for DiT-XL-2-512.

Baseline: generate 10 images (one per class) over 32 DDIM steps, recording:
  - initial noise (eps)
  - predicted clean latent (x0) after every step

Perturbation experiment: at each "injection step" that is a multiple of 4
(steps 4, 8, 12, ..., 32), we:
  1. Take the baseline x0 at that step.
  2. Apply spatial compression (downsample → upsample) at a given target size.
  3. Mix the compressed x0 back with the original noise at the correct noise level
     to reconstruct a noisy latent at that timestep.
  4. Continue denoising from that point using every step, every 2nd step, or
     every 4th step (stride variants).
  5. Measure MSE between the perturbed trajectory and the baseline trajectory
     at each subsequent step.

Three independent variables:
  1. Compression size  : [128, 256]  (target spatial size for down→up)
  2. Injection timing  : multiples of 4 across the 32 steps
  3. Step stride       : 1, 2, 4  (how many baseline steps are skipped per model call)

Results are saved as a pickle file containing a nested dict:
  results[compression_size][injection_step][stride] = {
      "divergence": list[float],   # per-step MSE relative to baseline
      "step_indices": list[int],   # which step indices the divergence was measured at
  }

Images from the baseline run are also saved for visual inspection.
"""

import os
import pickle
import zipfile

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32

from models import encode_prompt, load_model, predict, upsample_latents


def downsample_latents(latents: torch.Tensor, size: int) -> torch.Tensor:
    """Average-pool latents [B, C, H, W] to the given square spatial size."""
    if latents.shape[2] == size and latents.shape[3] == size:
        return latents
    return F.adaptive_avg_pool2d(latents, (size, size))

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_CLASSES = 10
N_IMAGES_PER_CLASS = 8
N_STEPS = 32
CFG_SCALE = 6.0
SEED = 42

# 8 images per class, 10 evenly-spaced ImageNet classes → batch size 80
_BASE_CLASSES = [i * (1000 // N_CLASSES) for i in range(N_CLASSES)]
CLASS_LABELS = [cls for cls in _BASE_CLASSES for _ in range(N_IMAGES_PER_CLASS)]

# Compression sizes to test (latent spatial dims: 128//8=16, 256//8=32)
COMPRESSION_SIZES = [128, 256]  # pixel space sizes

# Injection steps: every 4th step (1-indexed so step 4 = index 3, etc.)
INJECTION_STEPS = list(range(4, N_STEPS + 1, 4))  # [4, 8, 12, ..., 32]

# Step strides for the resumed trajectory
STRIDES = [1, 2, 4]

OUT_DIR = "results/trajectory_divergence"
RESULTS_PATH = os.path.join(OUT_DIR, "results.pkl")
ZIP_PATH = "results/trajectory_divergence.zip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_alpha(pipe, t: torch.Tensor) -> torch.Tensor:
    """Return sqrt(alpha_cumprod) at timestep t as a [1,1,1,1] tensor."""
    ac = pipe.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    return ac[t].sqrt().view(1, 1, 1, 1)


def decode_latents(pipe, latents: torch.Tensor) -> list[Image.Image]:
    latents = latents / pipe.vae.config.scaling_factor
    images = pipe.vae.decode(latents, return_dict=False)[0]
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(img) for img in images]


# ---------------------------------------------------------------------------
# Baseline trajectory recording
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_baseline(pipe, timesteps: torch.Tensor) -> dict:
    """
    Run the full 32-step baseline for all N_CLASSES images.

    Returns:
        {
            "eps":      [B, 4, 64, 64]  initial noise
            "x0s":      list of [B, 4, 64, 64] tensors, one per step (length N_STEPS)
            "latents":  list of [B, 4, 64, 64] noisy latents entering each step
            "timesteps": the timestep tensor used
            "final_latents": [B, 4, 64, 64] fully denoised latent
        }
    """
    B = N_CLASSES
    prompt_data = encode_prompt(pipe, "dit", CLASS_LABELS, device, dtype)
    latent_size = 64  # 512 // 8

    eps = torch.randn(B, 4, latent_size, latent_size, device=device, dtype=dtype)

    t0 = timesteps[0]
    a0 = get_alpha(pipe, t0)
    latents = a0 * torch.zeros_like(eps) + (1 - a0 ** 2).sqrt() * eps

    x0s = []
    noisy_latents = []

    for step_idx, t in enumerate(tqdm(timesteps, desc="Baseline")):
        a_t = get_alpha(pipe, t)
        noisy_latents.append(latents.clone())

        noise_pred = predict(pipe, "dit", latents, t, prompt_data, guidance_scale=CFG_SCALE)
        x0 = (latents - (1 - a_t ** 2).sqrt() * noise_pred) / a_t
        x0s.append(x0.clone())

        if step_idx + 1 < len(timesteps):
            t_next = timesteps[step_idx + 1]
            a_next = get_alpha(pipe, t_next)
        else:
            a_next = torch.ones(1, device=device, dtype=dtype).view(1, 1, 1, 1)

        latents = a_next * x0 + (1 - a_next ** 2).sqrt() * eps

    return {
        "eps": eps,
        "x0s": x0s,
        "noisy_latents": noisy_latents,
        "timesteps": timesteps,
        "final_latents": latents,
    }


# ---------------------------------------------------------------------------
# Perturbed trajectory
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_perturbed(
    pipe,
    baseline: dict,
    injection_step_idx: int,  # 0-indexed into timesteps
    compression_size: int,    # pixel size to compress to, e.g. 128 or 256
    stride: int,              # step stride for resumed denoising (1, 2, or 4)
) -> dict:
    """
    Inject a compressed x0 at `injection_step_idx` and continue denoising.

    Compression: x0 at injection step is downsampled to (compression_size//8)
    latent spatial resolution and upsampled back to 64 before mixing with noise.

    stride: controls which timesteps are used after injection.
      stride=1: use every subsequent timestep from the baseline schedule
      stride=2: use every 2nd
      stride=4: use every 4th

    Returns:
        {
            "x0s":          list of perturbed x0 tensors at the steps taken
            "step_indices": list of step indices (into baseline timesteps)
            "divergence":   list of per-step MSE vs baseline x0
        }
    """
    B = N_CLASSES
    prompt_data = encode_prompt(pipe, "dit", CLASS_LABELS, device, dtype)
    latent_size = 64
    comp_lat_size = compression_size // 8

    timesteps = baseline["timesteps"]
    eps = baseline["eps"]

    # Get x0 at injection step, compress it
    x0_inject = baseline["x0s"][injection_step_idx]  # [B, 4, 64, 64]
    x0_compressed = upsample_latents(
        downsample_latents(x0_inject, comp_lat_size), latent_size
    )

    # Reconstruct noisy latent at injection timestep using compressed x0 + original noise
    t_inj = timesteps[injection_step_idx]
    a_inj = get_alpha(pipe, t_inj)
    latents = a_inj * x0_compressed + (1 - a_inj ** 2).sqrt() * eps

    # Build the list of remaining step indices according to stride
    remaining_indices = list(range(injection_step_idx, len(timesteps), stride))

    perturbed_x0s = []
    step_indices = []
    divergences = []

    for i, step_idx in enumerate(remaining_indices):
        t = timesteps[step_idx]
        a_t = get_alpha(pipe, t)

        noise_pred = predict(pipe, "dit", latents, t, prompt_data, guidance_scale=CFG_SCALE)
        x0 = (latents - (1 - a_t ** 2).sqrt() * noise_pred) / a_t

        perturbed_x0s.append(x0.clone())
        step_indices.append(step_idx)

        # Divergence: MSE vs baseline x0 at the same step
        baseline_x0 = baseline["x0s"][step_idx]
        mse = F.mse_loss(x0.float(), baseline_x0.float()).item()
        divergences.append(mse)

        # Step to next point in stride schedule
        if i + 1 < len(remaining_indices):
            next_step_idx = remaining_indices[i + 1]
            t_next = timesteps[next_step_idx]
            a_next = get_alpha(pipe, t_next)
        else:
            a_next = torch.ones(1, device=device, dtype=dtype).view(1, 1, 1, 1)

        latents = a_next * x0 + (1 - a_next ** 2).sqrt() * eps

    return {
        "x0s": perturbed_x0s,
        "step_indices": step_indices,
        "divergence": divergences,
        "final_latents": latents,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    pipe = load_model("dit", device, dtype)
    pipe.scheduler.set_timesteps(N_STEPS)
    timesteps = pipe.scheduler.timesteps  # descending, length N_STEPS

    print(f"DDIM timesteps ({N_STEPS} steps): {timesteps.tolist()}")
    print(f"Classes: {CLASS_LABELS}")
    print(f"Injection steps (1-indexed): {INJECTION_STEPS}")
    print(f"Compression sizes: {COMPRESSION_SIZES}")
    print(f"Strides: {STRIDES}\n")

    # --- Baseline ---
    print("=== Running baseline ===")
    torch.manual_seed(SEED)
    baseline = run_baseline(pipe, timesteps)

    # Save baseline images
    baseline_dir = os.path.join(OUT_DIR, "baseline_images")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_imgs = decode_latents(pipe, baseline["final_latents"])
    for i, (img, cls) in enumerate(zip(baseline_imgs, CLASS_LABELS)):
        within_class_idx = i % N_IMAGES_PER_CLASS
        img.save(os.path.join(baseline_dir, f"class{cls:04d}_{within_class_idx:02d}.png"))
    print(f"Baseline images saved to {baseline_dir}\n")

    # --- Perturbation experiments ---
    # results[comp_size][inj_step][stride] = {"divergence": [...], "step_indices": [...]}
    results: dict = {}

    total = len(COMPRESSION_SIZES) * len(INJECTION_STEPS) * len(STRIDES)
    pbar = tqdm(total=total, desc="Experiments")

    for comp_size in COMPRESSION_SIZES:
        results[comp_size] = {}
        for inj_step in INJECTION_STEPS:
            inj_idx = inj_step - 1  # convert 1-indexed step to 0-indexed
            results[comp_size][inj_step] = {}
            for stride in STRIDES:
                torch.manual_seed(SEED)  # keep noise fixed across conditions
                out = run_perturbed(
                    pipe, baseline,
                    injection_step_idx=inj_idx,
                    compression_size=comp_size,
                    stride=stride,
                )
                results[comp_size][inj_step][stride] = {
                    "divergence": out["divergence"],
                    "step_indices": out["step_indices"],
                }

                # Save final images for this condition
                cond_dir = os.path.join(
                    OUT_DIR, f"comp{comp_size}_step{inj_step:02d}_stride{stride}"
                )
                os.makedirs(cond_dir, exist_ok=True)
                imgs = decode_latents(pipe, out["final_latents"])
                for i, (img, cls) in enumerate(zip(imgs, CLASS_LABELS)):
                    within_class_idx = i % N_IMAGES_PER_CLASS
                    img.save(os.path.join(cond_dir, f"class{cls:04d}_{within_class_idx:02d}.png"))

                pbar.set_postfix(comp=comp_size, step=inj_step, stride=stride)
                pbar.update(1)

    pbar.close()

    # Save results pickle
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Print a quick summary table
    print("\n=== Divergence summary (final-step MSE) ===")
    print(f"{'comp':>6} {'inj':>5} {'stride':>7}  final_mse")
    for comp_size in COMPRESSION_SIZES:
        for inj_step in INJECTION_STEPS:
            for stride in STRIDES:
                d = results[comp_size][inj_step][stride]["divergence"]
                final_mse = d[-1] if d else float("nan")
                print(f"{comp_size:>6} {inj_step:>5} {stride:>7}  {final_mse:.6f}")

    # Bundle images and metrics into a zip
    print(f"\nBundling results into {ZIP_PATH} ...")
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as zf:
        # Results pickle
        zf.write(RESULTS_PATH, arcname="results.pkl")
        # All image directories under OUT_DIR
        for dirpath, _, filenames in os.walk(OUT_DIR):
            for fname in sorted(filenames):
                if fname.endswith(".png"):
                    full_path = os.path.join(dirpath, fname)
                    arcname = os.path.relpath(full_path, start=os.path.dirname(OUT_DIR))
                    zf.write(full_path, arcname=arcname)
    print(f"Done. Zip written to {ZIP_PATH}")


if __name__ == "__main__":
    main()
