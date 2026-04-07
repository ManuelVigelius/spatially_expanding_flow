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

from models import encode_prompt, encode_image, predict, load_model, downsample_latents, upsample_latents

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
from config import device, dtype

num_inference_steps = 50
num_samples = 64
batch_size = 32
t_indices = torch.linspace(0, num_inference_steps - 1, 20).long()

# All multiples of 16 from 16 to 512
image_sizes = list(range(16, 513, 16))

SCALE_SIGMA_BY_SIZE = True
RESULTS_PATH = (
    "resize_experiment_results_scaled_sigma.pkl"
    if SCALE_SIGMA_BY_SIZE
    else "resize_experiment_results.pkl"
)


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


def _encode_samples(pipe, samples):
    """VAE-encode a list of samples at 512, return ref latents batched."""
    images = torch.cat([_process_sample(s, 512) for s in samples], dim=0)
    return encode_image(pipe, "sd3", images)


def run_resize_experiment(pipe, dataset_samples):
    """
    For every image size and every timestep, downsample 512 latents to `size`,
    add noise, run SD3, and record:
      - velocity MSE          (in latent space, at native size)
      - latent MSE            (predicted latent vs. ground-truth latent, both at native size)
      - upsampled_latent MSE  (predicted latent bilinearly upsampled to 512-latent size vs. ground-truth latents at 512)

    Samples are processed in batches of `batch_size` to utilise the GPU better.

    Returns:
        dict with keys "velocity", "latent", "upsampled_latent", each mapping
        size -> t_idx -> list[float]
    """
    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe.transformer.eval()

    vel_results = defaultdict(lambda: defaultdict(list))
    lat_results = defaultdict(lambda: defaultdict(list))
    upl_results = defaultdict(lambda: defaultdict(list))

    batches = [
        dataset_samples[i : i + batch_size]
        for i in range(0, len(dataset_samples), batch_size)
    ]

    # Encode empty prompt once — same for all samples/sizes/timesteps
    with torch.no_grad():
        prompt_data = encode_prompt(pipe, "sd3", "", device, dtype)

    for size in tqdm(image_sizes, desc="SD3 resize experiment"):
        with torch.no_grad():
            for batch in batches:
                B = len(batch)

                ref_latents = _encode_samples(pipe, batch)
                latent_spatial = ref_latents.shape[2:]  # (64, 64)

                latents = downsample_latents(ref_latents, size // 8)

                for t_idx in t_indices:
                    t_idx_int = t_idx.item()
                    sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=dtype)
                    if SCALE_SIGMA_BY_SIZE:
                        r = size / 512
                        sigma = (r * sigma) / (r * sigma + (1 - sigma))
                    timestep = (sigma * 1000).expand(B).to(device)

                    noise = torch.randn_like(latents)
                    noisy = (1 - sigma) * latents + sigma * noise
                    target = noise - latents

                    vel = predict(pipe, "sd3", noisy, timestep[0], prompt_data)

                    latent_pred = noisy - vel * sigma
                    for b in range(B):
                        vel_results[size][t_idx_int].append(
                            F.mse_loss(vel[b], target[b]).item()
                        )
                        lat_results[size][t_idx_int].append(
                            F.mse_loss(latent_pred[b], latents[b]).item()
                        )

                    upsampled_pred = upsample_latents(latent_pred, latent_spatial)
                    for b in range(B):
                        upl_results[size][t_idx_int].append(
                            F.mse_loss(upsampled_pred[b], ref_latents[b]).item()
                        )

    return {"velocity": vel_results, "latent": lat_results, "upsampled_latent": upl_results}


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
        if isinstance(d, (dict, defaultdict)):
            return {k: to_dict(v) for k, v in d.items()}
        return d

    results = to_dict(results)

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
