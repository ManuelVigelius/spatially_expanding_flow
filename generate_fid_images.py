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

import numpy as np
import torch
import torch.nn.functional as F
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

# If True, the model always runs at 512px but latents are blurred via downsample→upsample
# before each forward pass (virtual resize). The x0 prediction and re-noising still
# happen at full 512-latent resolution.
# If False, the model runs at the target resolution (actual resize).
VIRTUAL_RESIZE = False

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


def get_alpha(pipe, t: torch.Tensor) -> torch.Tensor:
    """Return sqrt(alpha_cumprod) at timestep t as a [1,1,1,1] tensor."""
    ac = pipe.scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    return ac[t].sqrt().view(1, 1, 1, 1)


@torch.no_grad()
def generate_batch(
    pipe,
    class_labels: list[int],
    size_schedule: list[int],
    timesteps: torch.Tensor,
) -> list[Image.Image]:
    """
    Generate one batch of images using the given per-step size schedule.

    At each step we:
      1. Run the model at the target (possibly smaller) resolution to predict x0.
      2. Upsample x0 to 512-latent space — clean signal only, no noise distortion.
      3. Re-noise the upsampled x0 to t_prev using the fixed 512-latent noise vector,
         preserving the correct noise statistics for the next step.

    size_schedule: list of pixel sizes, one per denoising step (length = N_STEPS).
    timesteps: 1-D tensor of DDPM timestep indices, descending (noisy->clean).
    """
    B = len(class_labels)
    prompt_data = encode_prompt(pipe, "dit", class_labels, device, dtype)

    latent_size = 512 // 8  # 64x64

    # Fixed noise vector, always kept at full 512-latent resolution
    eps = torch.randn(B, 4, latent_size, latent_size, device=device, dtype=dtype)

    # Bootstrap: noise a zero x0 to t=timesteps[0] to get the starting noisy latent
    t0 = timesteps[0]
    a0 = get_alpha(pipe, t0)
    latents = a0 * torch.zeros_like(eps) + (1 - a0 ** 2).sqrt() * eps  # = sqrt(1-alpha) * eps

    for step_idx, (t, size) in enumerate(zip(timesteps, size_schedule)):
        lat_size = size // 8
        a_t = get_alpha(pipe, t)

        if VIRTUAL_RESIZE:
            # Model always runs at 512px on a blurred (downsample→upsample) latent
            latents_input = upsample_latents(downsample_latents(latents, lat_size), latent_size)
            noise_pred = predict(pipe, "dit", latents_input, t, prompt_data, guidance_scale=CFG_SCALE)
            x0_full = (latents_input - (1 - a_t ** 2).sqrt() * noise_pred) / a_t
        else:
            # Model runs at target resolution; x0 is upsampled back to 512 afterwards
            latents_small = downsample_latents(latents, lat_size)
            noise_pred = predict(pipe, "dit", latents_small, t, prompt_data, guidance_scale=CFG_SCALE)
            x0_small = (latents_small - (1 - a_t ** 2).sqrt() * noise_pred) / a_t
            x0_full = upsample_latents(x0_small, latent_size)

        # Re-noise x0_full to t_prev using the fixed full-res noise vector
        if step_idx + 1 < len(timesteps):
            t_prev = timesteps[step_idx + 1]
            a_prev = get_alpha(pipe, t_prev)
        else:
            a_prev = torch.ones(1, device=device, dtype=dtype).view(1, 1, 1, 1)  # t=0, fully clean

        latents = a_prev * x0_full + (1 - a_prev ** 2).sqrt() * eps

    return decode_latents(pipe, latents)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)

    # Load schedules
    with open(SCHEDULES_PATH) as f:
        data = json.load(f)
    t_values = data["t_values"]          # list of t_idx values (length = N_STEPS_ORIG)
    optimal_sizes = data["schedules"][str(BUDGET_PCT)]  # one size per original step

    # We run DDIM with N_STEPS steps; use the scheduler to get evenly-spaced timesteps.
    pipe = load_model("dit", device, dtype)

    # DiT already uses DDIMScheduler; set_timesteps is called per-batch in the loop.
    # Do one set_timesteps here just to determine the timestep-to-size mapping.
    pipe.scheduler.set_timesteps(N_STEPS)

    # t_values are indices into the 50-step scheduler (0=clean, 49=most noisy).
    # optimal_sizes is stored in the same ascending order (index 0 = clean step).
    # DDIM timesteps are DDPM-scale integers descending (noisy→clean, e.g. 900→0).
    # Convert each DDIM timestep to a 0-49 index and look up the size directly.
    t_values_arr = np.array(t_values)                   # ascending 0..49
    max_t_value = float(t_values_arr.max())             # 49
    max_ddim_t = float(pipe.scheduler.timesteps[0])     # e.g. 900 for 10 steps
    ddim_ts = pipe.scheduler.timesteps.cpu().numpy()    # descending, length N_STEPS
    size_schedule = []
    for t in ddim_ts:
        t_idx = round(t / max_ddim_t * max_t_value)
        nearest = int(np.argmin(np.abs(t_values_arr - t_idx)))
        size_schedule.append(optimal_sizes[nearest])

    full_size_schedule = [512] * N_STEPS

    print(f"Optimal size schedule ({BUDGET_PCT}% budget): {size_schedule}")
    print(f"DDIM timesteps: {ddim_ts.tolist()}")

    mode = "virtual" if VIRTUAL_RESIZE else "actual"
    conditions = {
        "full": full_size_schedule,
        f"optimal_{mode}": size_schedule,
    }

    # 10 images per class, 1000 classes = 10k images total
    all_classes = [c for c in range(1000) for _ in range(10)]

    for cond_name, sched in conditions.items():
        out_dir = os.path.join(OUT_DIR, cond_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nGenerating {N_IMAGES} images [{cond_name}] -> {out_dir}")

        torch.manual_seed(SEED)  # same noise for both conditions
        img_idx = 0
        for start in tqdm(range(0, N_IMAGES, BATCH_SIZE)):
            batch_classes = all_classes[start : start + BATCH_SIZE]
            images = generate_batch(pipe, batch_classes, sched, pipe.scheduler.timesteps)
            for img in images:
                img.save(os.path.join(out_dir, f"{img_idx:05d}.png"))
                img_idx += 1

    print("\nDone.")


if __name__ == "__main__":
    main()
