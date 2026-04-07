"""
Experiment: Virtual resize vs real resize comparison for SD3.

For each image and timestep, we run SD3 under two conditions and compare
the predicted clean latents both get upsampled to 512x512:

  - Real resize:    image is actually resized to `size x size`, VAE-encoded,
                    noised, passed through SD3, then the predicted latent is
                    bilinearly upsampled to the 512 latent grid.

  - Virtual resize: image is encoded at 512x512, the clean latent is
                    spatially compressed (downsample+upsample via bilinear
                    interpolation to size x size and back), then noised and
                    passed through SD3 at 512x512.

Comparing the two predicted-clean-latents (both at 512x512) directly shows
how faithfully virtual resize approximates real resize.

Run with: python virtual_vs_real_resize_experiment.py
Results are saved to virtual_vs_real_resize_experiment_results.pkl
"""

import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from models import encode_prompt, encode_image, predict, load_model, downsample_latents, upsample_latents

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
from config import device, dtype

num_inference_steps = 50
num_samples = 64
batch_size = 16
t_indices = torch.linspace(0, num_inference_steps - 1, 20).long()

FULL_SIZE = 1024  # native resolution for this experiment
FULL_LATENT_SIZE = FULL_SIZE // 8  # 128

# Subset of sizes to keep runtime manageable
image_sizes = [64, 128, 256, 512, 768, 1024]

SCALE_SIGMA_BY_SIZE = True
RESULTS_PATH = "virtual_vs_real_resize_experiment_results_1024.pkl"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_tensor(image_pil, size):
    img = image_pil.convert("RGB").resize((size, size))
    t = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype) / 127.5 - 1.0


def _sigma_for_size(sigma_base, size):
    if SCALE_SIGMA_BY_SIZE:
        r = size / FULL_SIZE
        return (r * sigma_base) / (r * sigma_base + (1 - sigma_base))
    return sigma_base


def run_experiment(pipe, dataset_samples):
    """
    Returns a dict:
        results[size][t_idx] = list of per-sample dicts with keys:
            "real_vs_full"     - MSE(real_pred_upsampled, full_pred)
            "virtual_vs_full"  - MSE(virtual_pred, full_pred)
            "real_vs_virtual"  - MSE(real_pred_upsampled, virtual_pred)
    """
    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe.transformer.eval()

    # Encode empty prompt once (shared for all samples/sizes/timesteps)
    with torch.no_grad():
        prompt_data = encode_prompt(pipe, "sd3", "", device, dtype)

    results = {size: defaultdict(list) for size in image_sizes}

    batches = [
        dataset_samples[i: i + batch_size]
        for i in range(0, len(dataset_samples), batch_size)
    ]

    for size in tqdm(image_sizes, desc="sizes"):
        for batch in tqdm(batches, desc=f"  batches (size={size})", leave=False):
            B = len(batch)
            with torch.no_grad():
                # --- full resolution latents (shared reference) ---
                images_512 = torch.cat([_to_tensor(s["image"], FULL_SIZE) for s in batch], dim=0)
                assert images_512.shape == (B, 3, FULL_SIZE, FULL_SIZE), f"Expected images_512 {(B, 3, FULL_SIZE, FULL_SIZE)}, got {images_512.shape}"
                latents_512 = encode_image(pipe, "sd3", images_512)
                assert latents_512.shape == (B, 16, FULL_LATENT_SIZE, FULL_LATENT_SIZE), f"Expected latents_512 {(B, 16, FULL_LATENT_SIZE, FULL_LATENT_SIZE)}, got {latents_512.shape}"

                latent_spatial = latents_512.shape[2:]  # (FULL_LATENT_SIZE, FULL_LATENT_SIZE)

                latent_small_h = size // 8

                # --- real resize: downsample full-res latents to `size` ---
                latents_small = downsample_latents(latents_512, latent_small_h)

                # --- virtual resize: compress 512 latents to `size` and back ---
                latents_virtual = upsample_latents(downsample_latents(latents_512, latent_small_h), latent_spatial)

                for t_idx in t_indices:
                    t_idx_int = t_idx.item()
                    sigma_base = pipe.scheduler.sigmas[t_idx_int].to(dtype=dtype)
                    timestep = pipe.scheduler.timesteps[t_idx_int]

                    # Shared noise for all three conditions so variance cancels
                    noise_small = torch.randn_like(latents_small)
                    noise_512 = torch.randn_like(latents_512)

                    # ---- real resize prediction ----
                    sigma_real = _sigma_for_size(sigma_base, size)
                    noisy_real = (1 - sigma_real) * latents_small + sigma_real * noise_small
                    assert noisy_real.shape == latents_small.shape, f"noisy_real shape mismatch: {noisy_real.shape}"
                    vel_real = predict(pipe, "sd3", noisy_real, sigma_real * 1000, prompt_data)
                    assert vel_real.shape == latents_small.shape, f"vel_real shape mismatch: {vel_real.shape}"
                    latent_pred_real = noisy_real - vel_real * sigma_real  # [B, C, h, w]
                    latent_pred_real_up = upsample_latents(latent_pred_real, latent_spatial)
                    assert latent_pred_real_up.shape == latents_512.shape, f"latent_pred_real_up shape mismatch: {latent_pred_real_up.shape}"

                    # ---- virtual resize prediction (runs at full res, no sigma scaling) ----
                    sigma_virtual = sigma_base
                    noisy_virtual = (1 - sigma_virtual) * latents_virtual + sigma_virtual * noise_512
                    assert noisy_virtual.shape == latents_512.shape, f"noisy_virtual shape mismatch: {noisy_virtual.shape}"
                    vel_virtual = predict(pipe, "sd3", noisy_virtual, timestep, prompt_data)
                    assert vel_virtual.shape == latents_512.shape, f"vel_virtual shape mismatch: {vel_virtual.shape}"
                    latent_pred_virtual = noisy_virtual - vel_virtual * sigma_virtual  # [B, C, 64, 64]
                    latent_pred_virtual = upsample_latents(
                        downsample_latents(latent_pred_virtual, latent_small_h), latent_spatial
                    )
                    assert latent_pred_virtual.shape == latents_512.shape, f"latent_pred_virtual shape mismatch: {latent_pred_virtual.shape}"

                    for b in range(B):
                        results[size][t_idx_int].append({
                            "real_vs_gt": F.mse_loss(latent_pred_real_up[b], latents_512[b]).item(),
                            "virtual_vs_gt": F.mse_loss(latent_pred_virtual[b], latents_512[b]).item(),
                            "real_vs_virtual": F.mse_loss(latent_pred_real_up[b], latent_pred_virtual[b]).item(),
                        })

    # Convert defaultdicts to regular dicts
    return {size: dict(v) for size, v in results.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Loading DIV2K dataset...")
    dataset = load_dataset("mAiello00/DIV2K", split="train", streaming=True)
    dataset_samples = [{"image": s["image"].convert("RGB")} for s in dataset.take(num_samples)]

    print("Loading SD3...")
    pipe = load_model("sd3", device, dtype)

    print("Running experiment...")
    results = run_experiment(pipe, dataset_samples)

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
