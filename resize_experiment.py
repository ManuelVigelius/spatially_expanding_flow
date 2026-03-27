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


def _encode_samples(pipe, samples, size, sf, sc):
    """VAE-encode a list of samples at `size`, return (latents, ref_pixels) batched."""
    images = torch.cat([_process_sample(s, size) for s in samples], dim=0)
    latents = (pipe.vae.encode(images).latent_dist.sample() - sf) * sc
    if size == 512:
        ref_pixels = pipe.vae.decode(latents / sc + sf).sample
    else:
        ref_pixels = None
    return latents, ref_pixels


def run_resize_experiment(pipe, dataset_samples):
    """
    For every image size and every timestep, encode a batch of images at that
    size, add noise, run SD3, and record:
      - velocity MSE   (in latent space)
      - latent MSE     (predicted latent vs. ground-truth latent, both at native size)
      - pixel MSE      (predicted image decoded and bilinearly upsampled to 512 vs. ground-truth image at 512)
      - upsampled_latent MSE  (predicted latent bilinearly upsampled to 512-latent size vs. ground-truth latents at 512)

    Samples are processed in batches of `batch_size` to utilise the GPU better.

    Returns:
        dict with keys "velocity", "latent", "pixel", "upsampled_latent", each mapping
        size -> t_idx -> list[float]
    """
    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe.transformer.eval()

    vel_results = defaultdict(lambda: defaultdict(list))
    lat_results = defaultdict(lambda: defaultdict(list))
    pix_results = defaultdict(lambda: defaultdict(list))
    upl_results = defaultdict(lambda: defaultdict(list))

    sf = pipe.vae.config.shift_factor
    sc = pipe.vae.config.scaling_factor

    batches = [
        dataset_samples[i : i + batch_size]
        for i in range(0, len(dataset_samples), batch_size)
    ]

    # Encode empty prompt once — same for all samples/sizes/timesteps
    with torch.no_grad():
        prompt_embeds_1, pooled_1 = encode_prompt_for_model(pipe, "sd3", "", device, dtype)

    for size in tqdm(image_sizes, desc="SD3 resize experiment"):
        with torch.no_grad():
            for batch in batches:
                B = len(batch)

                latents, ref_pixels = _encode_samples(pipe, batch, size, sf, sc)

                # Reference pixels and 512 latents; re-use across timesteps
                if ref_pixels is None:
                    images_512 = torch.cat([_process_sample(s, 512) for s in batch], dim=0)
                    ref_latents = (pipe.vae.encode(images_512).latent_dist.sample() - sf) * sc
                    ref_pixels = pipe.vae.decode(ref_latents / sc + sf).sample  # [B, 3, 512, 512]
                else:
                    # size == 512: latents are already at full resolution
                    ref_latents = latents

                # Tile prompt embeddings to match batch size
                prompt_embeds = prompt_embeds_1.expand(B, -1, -1)
                pooled = pooled_1.expand(B, -1)

                for t_idx in t_indices:
                    t_idx_int = t_idx.item()
                    sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=dtype)
                    if SCALE_SIGMA_BY_SIZE:
                        r = size / 512
                        sigma = (r * sigma) / (r * sigma + (1 - sigma))
                    timestep = pipe.scheduler.timesteps[t_idx_int]

                    noise = torch.randn_like(latents)
                    noisy = (1 - sigma) * latents + sigma * noise
                    target = noise - latents

                    vel = pipe.transformer(
                        hidden_states=noisy,
                        timestep=timestep.expand(B).to(device),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled,
                        return_dict=False,
                    )[0]

                    latent_pred = noisy - vel * sigma
                    noise_pred = noisy + vel * (1 - sigma)

                    # Per-sample MSE (reduce over all dims except batch)
                    for b in range(B):
                        vel_results[size][t_idx_int].append(
                            F.mse_loss(noise_pred[b], target[b]).item()
                        )
                        lat_results[size][t_idx_int].append(
                            F.mse_loss(latent_pred[b], latents[b]).item()
                        )

                    pred_pixels = pipe.vae.decode(latent_pred / sc + sf).sample
                    if size != 512:
                        pred_pixels = F.interpolate(
                            pred_pixels, size=(512, 512), mode="bilinear", align_corners=False
                        )
                    for b in range(B):
                        pix_results[size][t_idx_int].append(
                            F.mse_loss(pred_pixels[b], ref_pixels[b]).item()
                        )

                    latent_spatial = ref_latents.shape[2:]
                    if size != 512:
                        upsampled_pred = F.interpolate(
                            latent_pred, size=latent_spatial, mode="bilinear", align_corners=False
                        )
                    else:
                        upsampled_pred = latent_pred
                    for b in range(B):
                        upl_results[size][t_idx_int].append(
                            F.mse_loss(upsampled_pred[b], ref_latents[b]).item()
                        )

    return {"velocity": vel_results, "latent": lat_results, "pixel": pix_results, "upsampled_latent": upl_results}


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
