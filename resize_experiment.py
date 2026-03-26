"""
Experiment: SD3 performance across image sizes and noise scales.

Tests how well SD3 predicts the velocity regression target and clean latents
when the input image is resized to different resolutions (all multiples of 8
up to 512) across 20 evenly-spaced noise timesteps.

Run with: python resize_experiment.py
Results are saved to resize_experiment_results.pkl
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
t_indices = torch.linspace(0, num_inference_steps - 1, 20).long()

# All multiples of 8 from 8 to 512
image_sizes = list(range(8, 513, 8))

RESULTS_PATH = "resize_experiment_results.pkl"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _process_sample(sample, size):
    """Return a normalised image tensor at the given square resolution."""
    image = sample["image"].convert("RGB").resize((size, size))
    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device=device, dtype=dtype)
    image = (image / 127.5) - 1.0
    return image


def run_resize_experiment(pipe, dataset_samples):
    """
    For every image size and every timestep, encode the image at that size,
    add noise, run SD3, and record:
      - velocity MSE  (in latent space)
      - latent MSE    (predicted latent vs. ground-truth latent, both at native size)
      - image MSE     (predicted image bilinearly upsampled to 512 vs. ground-truth image at 512)

    Returns:
        dict with keys "velocity", "latent", "image", each mapping
        size -> t_idx -> list[float]
    """
    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe.transformer.eval()

    vel_results = defaultdict(lambda: defaultdict(list))
    lat_results = defaultdict(lambda: defaultdict(list))
    img_results = defaultdict(lambda: defaultdict(list))

    # VAE scale/shift helpers
    sf = pipe.vae.config.shift_factor
    sc = pipe.vae.config.scaling_factor

    for sample in tqdm(dataset_samples, desc="SD3 resize experiment"):
        with torch.no_grad():
            # Ground-truth image and decoded pixels at full 512x512 resolution
            image_512 = _process_sample(sample, 512)
            raw_512 = pipe.vae.encode(image_512).latent_dist.sample()
            # Decode reference once: used for image-space MSE
            ref_pixels = pipe.vae.decode((raw_512 - sf) * sc).sample  # [1, 3, 512, 512]

            # Encode empty prompt once per sample (same for all sizes/timesteps)
            prompt_embeds, pooled = encode_prompt_for_model(
                pipe, "sd3", "", device, dtype
            )

            for size in image_sizes:
                image = _process_sample(sample, size)

                raw_latents = pipe.vae.encode(image).latent_dist.sample()
                latents = (raw_latents - sf) * sc

                for t_idx in t_indices:
                    t_idx_int = t_idx.item()
                    sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=dtype)
                    timestep = pipe.scheduler.timesteps[t_idx_int]

                    noise = torch.randn_like(latents)
                    noisy = (1 - sigma) * latents + sigma * noise
                    target = noise - latents

                    vel = pipe.transformer(
                        hidden_states=noisy,
                        timestep=timestep.unsqueeze(0).to(device),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=False,
                    )[0]

                    latent_pred = noisy - vel * sigma
                    noise_pred = noisy + vel * (1 - sigma)

                    vel_results[size][t_idx_int].append(
                        F.mse_loss(noise_pred, target).item()
                    )
                    lat_results[size][t_idx_int].append(
                        F.mse_loss(latent_pred, latents).item()
                    )

                    # Decode predicted latents, upsample to 512, compare to reference
                    pred_pixels = pipe.vae.decode(latent_pred / sc + sf).sample
                    if size != 512:
                        pred_pixels = F.interpolate(
                            pred_pixels, size=(512, 512), mode="bilinear", align_corners=False
                        )
                    img_results[size][t_idx_int].append(
                        F.mse_loss(pred_pixels, ref_pixels).item()
                    )

    return {"velocity": vel_results, "latent": lat_results, "image": img_results}


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
    results = run_resize_experiment(pipe, dataset_samples)

    # Convert defaultdicts to regular dicts for pickling
    def to_dict(d):
        if isinstance(d, defaultdict):
            return {k: to_dict(v) for k, v in d.items()}
        return d

    results = to_dict(results)

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
