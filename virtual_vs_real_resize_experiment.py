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

from models import encode_prompt_for_model, load_model

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32

num_inference_steps = 50
num_samples = 64
batch_size = 16
t_indices = torch.linspace(0, num_inference_steps - 1, 20).long()

# Subset of sizes to keep runtime manageable
image_sizes = [32, 64, 128, 256, 384, 512]

SCALE_SIGMA_BY_SIZE = True
RESULTS_PATH = "virtual_vs_real_resize_experiment_results.pkl"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_tensor(image_pil, size):
    img = image_pil.convert("RGB").resize((size, size))
    t = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype) / 127.5 - 1.0


def _sigma_for_size(sigma_base, size):
    if SCALE_SIGMA_BY_SIZE:
        r = size / 512
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

    sf = pipe.vae.config.shift_factor
    sc = pipe.vae.config.scaling_factor

    # Encode prompts once (empty prompt for all)
    with torch.no_grad():
        prompt_embeds_1, pooled_1 = encode_prompt_for_model(pipe, "sd3", "", device, dtype)

    results = {size: defaultdict(list) for size in image_sizes}

    batches = [
        dataset_samples[i: i + batch_size]
        for i in range(0, len(dataset_samples), batch_size)
    ]

    for size in tqdm(image_sizes, desc="sizes"):
        for batch in tqdm(batches, desc=f"  batches (size={size})", leave=False):
            B = len(batch)
            prompt_embeds = prompt_embeds_1.expand(B, -1, -1)
            pooled = pooled_1.expand(B, -1)

            with torch.no_grad():
                # --- full 512 latents (shared reference) ---
                images_512 = torch.cat([_to_tensor(s["image"], 512) for s in batch], dim=0)
                latents_512 = (pipe.vae.encode(images_512).latent_dist.sample() - sf) * sc  # [B, C, 64, 64]

                latent_spatial = latents_512.shape[2:]  # (64, 64)

                if size < 512:
                    # --- real resize: encode at `size` ---
                    images_small = torch.cat([_to_tensor(s["image"], size) for s in batch], dim=0)
                    latents_small = (pipe.vae.encode(images_small).latent_dist.sample() - sf) * sc

                    # --- virtual resize: compress 512 latents to `size` and back ---
                    latent_small_h = size // 8
                    latent_small_w = size // 8
                    latents_compressed = F.interpolate(
                        latents_512,
                        size=(latent_small_h, latent_small_w),
                        mode="bilinear",
                        align_corners=True,
                    )
                    latents_virtual = F.interpolate(
                        latents_compressed,
                        size=latent_spatial,
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    latents_small = latents_512
                    latents_virtual = latents_512

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
                    vel_real = pipe.transformer(
                        hidden_states=noisy_real,
                        timestep=timestep.expand(B).to(device),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=False,
                    )[0]
                    latent_pred_real = noisy_real - vel_real * sigma_real  # [B, C, h, w]
                    if size < 512:
                        latent_pred_real_up = F.interpolate(
                            latent_pred_real, size=latent_spatial, mode="bilinear", align_corners=True
                        )
                    else:
                        latent_pred_real_up = latent_pred_real

                    # ---- virtual resize prediction (runs at 512, compressed input) ----
                    sigma_virtual = _sigma_for_size(sigma_base, size)
                    noisy_virtual = (1 - sigma_virtual) * latents_virtual + sigma_virtual * noise_512
                    vel_virtual = pipe.transformer(
                        hidden_states=noisy_virtual,
                        timestep=timestep.expand(B).to(device),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=False,
                    )[0]
                    latent_pred_virtual = noisy_virtual - vel_virtual * sigma_virtual  # [B, C, 64, 64]
                    if size < 512:
                        latent_pred_virtual = F.interpolate(
                            F.interpolate(
                                latent_pred_virtual,
                                size=(latent_small_h, latent_small_w),
                                mode="bilinear",
                                align_corners=True,
                            ),
                            size=latent_spatial,
                            mode="bilinear",
                            align_corners=True,
                        )

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
    print("Loading COCO dataset...")
    dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)
    dataset_samples = list(dataset.take(num_samples))

    print("Loading SD3...")
    pipe = load_model("sd3", device, dtype)

    print("Running experiment...")
    results = run_experiment(pipe, dataset_samples)

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
