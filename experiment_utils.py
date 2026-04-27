"""Shared helpers for resize / virtual-resize / virtual-vs-real experiments."""

import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

DATASET_CONFIGS = {
    "coco":       {"path": "detection-datasets/coco",  "split": "val",        "image_field": "image", "label_field": None},
    "imagenet1k": {"path": "imagenet-1k",              "split": "validation", "image_field": "image", "label_field": "label"},
    "div2k":      {"path": "mAiello00/DIV2K",          "split": "train",      "image_field": "image", "label_field": None},
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_samples(dataset_name: str, n: int) -> list:
    """Load n samples from the given dataset.

    Returns a list of dicts with keys ``"image"`` (PIL) and ``"label"`` (int or None).
    """
    cfg = DATASET_CONFIGS[dataset_name]
    ds = load_dataset(cfg["path"], split=cfg["split"], streaming=True)
    raw = list(ds.take(n))
    return [
        {
            "image": s[cfg["image_field"]].convert("RGB"),
            "label": s[cfg["label_field"]] if cfg["label_field"] else None,
        }
        for s in raw
    ]


def image_to_tensor(image_pil, size: int, device, dtype):
    """Resize a PIL image to ``size x size`` and return a normalised [1, 3, H, W] tensor."""
    img = image_pil.resize((size, size))
    t = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype) / 127.5 - 1.0


# --------------------------------------------------------------------------- #
# Scheduler helpers
# --------------------------------------------------------------------------- #

def setup_scheduler(pipe, model_name: str, num_inference_steps: int, device) -> None:
    """Call set_timesteps on the scheduler, handling Flux's mu offset."""
    if model_name == "flux":
        sc = pipe.scheduler.config
        lh = pipe.transformer.config.sample_size  # latent height in tokens
        lw = lh
        ph, pw = lh // 2, lw // 2
        seq_len = ph * pw
        mu = sc.base_shift + sc.max_shift * (
            (seq_len - sc.base_image_seq_len) / (sc.max_image_seq_len - sc.base_image_seq_len)
        )
        pipe.scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    else:
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)


def is_flow_matching(scheduler) -> bool:
    """Return True when the scheduler uses flow-matching sigmas."""
    return hasattr(scheduler, "sigmas")


def get_sigmas(scheduler):
    """Return the 1-D sigma tensor (flow-matching schedulers only)."""
    return scheduler.sigmas


def get_alphas(scheduler, device, dtype):
    """Return (sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod) for DDPM/DDIM schedulers."""
    ac = scheduler.alphas_cumprod[scheduler.timesteps.cpu()].to(device=device, dtype=dtype)
    return ac.sqrt(), (1 - ac).sqrt()


# --------------------------------------------------------------------------- #
# Noise / noisy-latent helpers
# --------------------------------------------------------------------------- #

def make_noisy(latents, noise, sigma):
    """Flow-matching forward process: (1 - sigma) * latents + sigma * noise."""
    return (1 - sigma) * latents + sigma * noise


def scale_sigma(sigma, size: int, ref_size: int):
    """Scale sigma according to the resolution ratio.

    Formula: sigma_scaled = (r * sigma) / (r * sigma + (1 - sigma))
    where r = size / ref_size.
    """
    r = size / ref_size
    return (r * sigma) / (r * sigma + (1 - sigma))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def mse(a, b) -> float:
    """Per-sample MSE as a Python float."""
    return F.mse_loss(a, b).item()


# --------------------------------------------------------------------------- #
# Results helpers
# --------------------------------------------------------------------------- #

def nested_defaultdict():
    """Return a two-level defaultdict(list): d[key1][key2] -> list."""
    return defaultdict(lambda: defaultdict(list))


def defaultdict_to_dict(d):
    """Recursively convert defaultdicts to plain dicts for pickling."""
    if isinstance(d, (dict, defaultdict)):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def save_results(results: dict, path: str) -> None:
    """Pickle results to path."""
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {path}")


def decode_latents(pipe, latents: torch.Tensor):
    """VAE-decode SD3 latents [B, C, H, W] to a list of PIL images."""
    from PIL import Image
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.no_grad():
        images = pipe.vae.decode(latents, return_dict=False)[0]
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(img) for img in images]


# --------------------------------------------------------------------------- #
# Higher-level helpers
# --------------------------------------------------------------------------- #

def encode_batch(pipe, model_name, batch, ref_size, device, dtype):
    """VAE-encode a list of samples at ref_size, returning a [B, C, H, W] latent tensor."""
    from models import encode_image
    images = torch.cat([image_to_tensor(s["image"], ref_size, device, dtype) for s in batch], dim=0)
    return encode_image(pipe, model_name, images)


def denoise_step_flow(pipe, model_name, latents, noise, t_idx_int, prompt_data,
                      guidance_scale=1.0, scale_sigma_fn=None):
    """One flow-matching denoising step.

    Adds noise at the given scheduler index, runs the model, and recovers the
    predicted clean latent.

    Args:
        scale_sigma_fn: optional callable(sigma) -> sigma applied before noising,
                        e.g. ``lambda s: scale_sigma(s, size, ref_size)``.

    Returns:
        latent_pred [B, C, H, W], sigma (effective, after optional scaling)
    """
    from models import predict
    B = latents.shape[0]
    device = latents.device
    sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=latents.dtype)
    if scale_sigma_fn is not None:
        sigma = scale_sigma_fn(sigma)
    timestep = (sigma * 1000).expand(B).to(device)
    noisy = make_noisy(latents, noise, sigma)
    vel = predict(pipe, model_name, noisy, timestep[0], prompt_data, guidance_scale=guidance_scale)
    return noisy - vel * sigma, sigma


def denoise_step_ddpm(pipe, model_name, latents, noise, t_idx_int, sqrt_alphas,
                      sqrt_one_minus_alphas, prompt_data, guidance_scale=1.0):
    """One DDPM/DDIM denoising step.

    Returns:
        latent_pred [B, C, H, W]
    """
    from models import predict
    B = latents.shape[0]
    device = latents.device
    sa = sqrt_alphas[t_idx_int]
    sb = sqrt_one_minus_alphas[t_idx_int]
    timestep = pipe.scheduler.timesteps[t_idx_int].expand(B).to(device)
    noisy = sa * latents + sb * noise
    vel = predict(pipe, model_name, noisy, timestep[0], prompt_data, guidance_scale=guidance_scale)
    return (noisy - sb * vel) / sa
