"""
Generate 10k images with DiT-XL-2 using DDIM (10 steps) for FID evaluation.

Two conditions are compared:
  - full:    every denoising step runs at 512px
  - optimal: step sizes follow the 50%-budget optimal schedule from optimal_schedules.json

Classes are sampled uniformly at random from [0, 999] (ImageNet-1k).
CFG scale is 4.0 (null class = 1000).

Images are saved as PNGs under:
  results/fid_images/full/
  results/fid_images/optimal/
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from PIL import Image
from tqdm import tqdm

from config import device, dtype
from models import encode_prompt, load_model, predict, downsample_latents, upsample_latents

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_IMAGES = 10_000
BATCH_SIZE = 32
N_STEPS = 10
CFG_SCALE = 4.0
BUDGET_PCT = 50  # which schedule to use from optimal_schedules.json
SCHEDULES_PATH = "optimal_schedules.json"
OUT_DIR = "results/fid_images"
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_latents(pipe, latents: torch.Tensor) -> list[Image.Image]:
    """VAE-decode latents [B, C, H, W] to a list of PIL images."""
    latents = latents / pipe.vae.config.scaling_factor
    images = pipe.vae.decode(latents, return_dict=False)[0]  # [B, 3, H, W] in [-1, 1]
    images = (images.clamp(-1, 1) + 1) / 2  # [0, 1]
    images = (images * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(img) for img in images]


def ddim_step(
    pipe,
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    t_cur: torch.Tensor,
    t_prev: torch.Tensor,
) -> torch.Tensor:
    """Single DDIM step from t_cur -> t_prev using the scheduler's alphas_cumprod."""
    ac = pipe.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    alpha_cur = ac[t_cur].view(-1, 1, 1, 1)
    alpha_prev = ac[t_prev].view(-1, 1, 1, 1)
    # DDIM update (eta=0, deterministic)
    x0_pred = (latents - (1 - alpha_cur).sqrt() * noise_pred) / alpha_cur.sqrt()
    return alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred


@torch.no_grad()
def generate_batch(
    pipe,
    class_labels: list[int],
    size_schedule: list[int],
    timesteps: torch.Tensor,
) -> list[Image.Image]:
    """
    Generate one batch of images using the given per-step size schedule.

    size_schedule: list of pixel sizes, one per denoising step (length = N_STEPS).
    timesteps: 1-D tensor of scheduler timestep indices, length N_STEPS + 1
               (timesteps[0] is the noisiest, timesteps[-1] = 0 is clean).
    """
    B = len(class_labels)
    prompt_data = encode_prompt(pipe, "dit", class_labels, device, dtype)

    # Start from pure noise at 512px latent resolution (64x64 latent)
    latent_size = 512 // 8  # = 64
    latents = torch.randn(B, 4, latent_size, latent_size, device=device, dtype=dtype)

    for step_idx, size in enumerate(size_schedule):
        t_cur = timesteps[step_idx].expand(B).to(device)
        t_prev = timesteps[step_idx + 1].expand(B).to(device)

        lat_size = size // 8
        latents_small = downsample_latents(latents, lat_size)

        noise_pred = predict(pipe, "dit", latents_small, t_cur[0], prompt_data, guidance_scale=CFG_SCALE)

        # DDIM step at native size, then upsample back to 512 latent space
        latents_small = ddim_step(pipe, latents_small, noise_pred, t_cur[0], t_prev[0])
        latents = upsample_latents(latents_small, latent_size)

    return decode_latents(pipe, latents)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load schedules
    with open(SCHEDULES_PATH) as f:
        data = json.load(f)
    t_values = data["t_values"]          # list of t_idx values (length = N_STEPS_ORIG)
    optimal_sizes = data["schedules"][str(BUDGET_PCT)]  # one size per original step

    # We run DDIM with N_STEPS steps; use the scheduler to get evenly-spaced timesteps.
    pipe = load_model("dit", device, dtype)

    # Replace the scheduler with DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(N_STEPS)
    timesteps = pipe.scheduler.timesteps  # descending, e.g. [981, 871, ..., 21, 1]

    # Map each DDIM step to the nearest t_value in the optimal schedule to get
    # the corresponding image size.  t_values are ascending (0 = clean), timesteps
    # are DDPM-style indices (999 = most noisy).  DiT uses DDPM timesteps where
    # t=0 corresponds to clean; the scheduler timesteps already are in [0, 999].
    t_values_arr = np.array(t_values)
    ddim_ts = timesteps.cpu().numpy()  # length N_STEPS
    size_schedule = []
    for t in ddim_ts:
        nearest = int(np.argmin(np.abs(t_values_arr - t)))
        size_schedule.append(optimal_sizes[nearest])

    full_size_schedule = [512] * N_STEPS

    print(f"Optimal size schedule ({BUDGET_PCT}% budget): {size_schedule}")
    print(f"DDIM timesteps: {ddim_ts.tolist()}")

    # Append t=0 as the "previous" for the last step
    timesteps_with_end = torch.cat([timesteps, torch.zeros(1, dtype=timesteps.dtype)])

    conditions = {
        "full": full_size_schedule,
        "optimal": size_schedule,
    }

    # Sample class labels once so both conditions use identical classes/noise seeds
    all_classes = [random.randint(0, 999) for _ in range(N_IMAGES)]

    for cond_name, sched in conditions.items():
        out_dir = os.path.join(OUT_DIR, cond_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nGenerating {N_IMAGES} images [{cond_name}] -> {out_dir}")

        torch.manual_seed(SEED)  # same noise for both conditions
        img_idx = 0
        for start in tqdm(range(0, N_IMAGES, BATCH_SIZE)):
            batch_classes = all_classes[start : start + BATCH_SIZE]
            images = generate_batch(pipe, batch_classes, sched, timesteps_with_end)
            for img in images:
                img.save(os.path.join(out_dir, f"{img_idx:05d}.png"))
                img_idx += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
