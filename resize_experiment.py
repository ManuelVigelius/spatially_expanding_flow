"""
Experiment: model performance across image sizes and noise scales.

Tests how well a model predicts the velocity regression target and clean latents
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

MODEL = "dit"
DATASET = "imagenet1k"
SCALE_SIGMA_BY_SIZE = True
CFG_SCALE = 4.0
RESULTS_PATH = f"resize_experiment_{MODEL}_{DATASET}{'_scaled_sigma' if SCALE_SIGMA_BY_SIZE else ''}.pkl"

DATASET_CONFIGS = {
    "coco":       {"path": "detection-datasets/coco", "split": "val",        "image_field": "image", "label_field": None},
    "imagenet1k": {"path": "imagenet-1k",             "split": "validation", "image_field": "image", "label_field": "label"},
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_samples(dataset_name, n):
    """Load n samples from the given dataset and normalise to {"image", "label"}."""
    cfg = DATASET_CONFIGS[dataset_name]
    ds = load_dataset(cfg["path"], split=cfg["split"], streaming=True)
    raw = list(ds.take(n))
    return [
        {"image": s[cfg["image_field"]], "label": s[cfg["label_field"]] if cfg["label_field"] else None}
        for s in raw
    ]


def _is_flow_matching(scheduler):
    return hasattr(scheduler, "sigmas")


def _get_sigmas(scheduler):
    """Return a 1-D flow-matching sigma tensor (only valid for flow-matching schedulers)."""
    return scheduler.sigmas


def _get_alphas(scheduler):
    """Return sqrt_alpha and sqrt_one_minus_alpha at each scheduled timestep (DDPM/DDIM)."""
    ac = scheduler.alphas_cumprod[scheduler.timesteps.cpu()].to(device=device, dtype=dtype)
    return ac.sqrt(), (1 - ac).sqrt()


def _get_prompt(batch):
    """Return the prompt argument for encode_prompt given the current MODEL and batch."""
    if MODEL == "dit":
        return [s["label"] if s["label"] is not None else 1000 for s in batch]
    return ""


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
    return encode_image(pipe, MODEL, images)


def run_resize_experiment(pipe, dataset_samples):
    """
    For every image size and every timestep, downsample 512 latents to `size`,
    add noise, run the model, and record:
      - latent MSE            (predicted latent vs. ground-truth latent, both at native size)
      - upsampled_latent MSE  (predicted latent bilinearly upsampled to 512-latent size vs. ground-truth latents at 512)

    Samples are processed in batches of `batch_size` to utilise the GPU better.

    Returns:
        dict with keys "velocity", "latent", "upsampled_latent", each mapping
        size -> t_idx -> list[float]
    """
    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe.transformer.eval()
    flow_matching = _is_flow_matching(pipe.scheduler)
    if flow_matching:
        sigmas = _get_sigmas(pipe.scheduler)
    else:
        sqrt_alphas, sqrt_one_minus_alphas = _get_alphas(pipe.scheduler)

    lat_results = defaultdict(lambda: defaultdict(list))
    upl_results = defaultdict(lambda: defaultdict(list))

    batches = [
        dataset_samples[i : i + batch_size]
        for i in range(0, len(dataset_samples), batch_size)
    ]

    for size in tqdm(image_sizes, desc=f"{MODEL} resize experiment"):
        with torch.no_grad():
            for batch in batches:
                B = len(batch)

                # Encode prompt per-batch so DiT can use per-sample class labels
                prompt_data = encode_prompt(pipe, MODEL, _get_prompt(batch), device, dtype)

                ref_latents = _encode_samples(pipe, batch)
                latent_spatial = ref_latents.shape[2:]  # (64, 64)

                latents = downsample_latents(ref_latents, size // 8)

                for t_idx in t_indices:
                    t_idx_int = t_idx.item()
                    noise = torch.randn_like(latents)

                    if flow_matching:
                        sigma = sigmas[t_idx_int].to(dtype=dtype)
                        if SCALE_SIGMA_BY_SIZE:
                            r = size / 512
                            sigma = (r * sigma) / (r * sigma + (1 - sigma))
                        timestep = (sigma * 1000).expand(B).to(device)
                        noisy = (1 - sigma) * latents + sigma * noise
                        pred = predict(pipe, MODEL, noisy, timestep[0], prompt_data, guidance_scale=CFG_SCALE)
                        latent_pred = noisy - pred * sigma
                    else:
                        sa = sqrt_alphas[t_idx_int]
                        sb = sqrt_one_minus_alphas[t_idx_int]
                        timestep = pipe.scheduler.timesteps[t_idx_int].expand(B).to(device)
                        noisy = sa * latents + sb * noise
                        pred = predict(pipe, MODEL, noisy, timestep[0], prompt_data, guidance_scale=CFG_SCALE)
                        latent_pred = (noisy - sb * pred) / sa
                    for b in range(B):
                        lat_results[size][t_idx_int].append(
                            F.mse_loss(latent_pred[b], latents[b]).item()
                        )

                    upsampled_pred = upsample_latents(latent_pred, latent_spatial)
                    for b in range(B):
                        upl_results[size][t_idx_int].append(
                            F.mse_loss(upsampled_pred[b], ref_latents[b]).item()
                        )

    return {"latent": lat_results, "upsampled_latent": upl_results}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print(f"Loading {DATASET} dataset...")
    dataset_samples = load_samples(DATASET, num_samples)

    print(f"Loading {MODEL}...")
    pipe = load_model(MODEL, device, dtype)

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
