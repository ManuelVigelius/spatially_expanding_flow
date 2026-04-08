"""
Generate 10k images with DiT-XL-2 using DDIM (10 steps) for FID evaluation.

Five hand-written size schedules are compared, each run with both
virtual resize and actual resize, for a total of 10 conditions.

Schedules (10 steps, index 0 = noisiest, index 9 = cleanest):
  full        — all steps at 512px (baseline)
  early_small — 256px for first 5 noisy steps, then 512px
  late_small  — 512px for first 5 steps, then 256px for last 5
  gradual     — linearly ramp from 128px up to 512px
  aggressive  — 128px for first 7 noisy steps, then 512px for last 3

For each schedule, two resize modes are run:
  actual  — model runs at target resolution; x0 upsampled back to 512
  virtual — model always runs at 512px on a blurred (down→up) latent

Classes are sampled uniformly from [0, 999] (ImageNet-1k), 10 per class.
CFG scale is 4.0. All images saved as PNGs; results bundled into a zip.

Images are saved as PNGs under:
  results/fid_images/<schedule_name>_<mode>/
A final zip is written to:
  results/fid_images.zip
"""

import os
import zipfile

import torch
from PIL import Image
from tqdm import tqdm

from config import device, dtype
from models import encode_prompt, load_model, predict, downsample_latents, upsample_latents

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_IMAGES = 32
BATCH_SIZE = 32
N_STEPS = 10
CFG_SCALE = 4.0
OUT_DIR = "results/fid_images"
ZIP_PATH = "results/fid_images.zip"
SEED = 0

# If True, sample class labels uniformly at random (seeded by SEED).
# If False, use deterministic labels: 10 images per class, all 1000 classes.
RANDOM_CLASSES = True

# ---------------------------------------------------------------------------
# Hand-written schedules (pixel sizes, one per denoising step, noisiest first)
# ---------------------------------------------------------------------------
SCHEDULES: dict[str, list[int]] = {
    "full":           [512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
    "early_small":    [256, 256, 256, 256, 256, 512, 512, 512, 512, 512],
    "gradual_mild":   [256, 256, 320, 320, 384, 384, 448, 448, 512, 512],
    "gradual_medium": [192, 192, 256, 256, 320, 384, 448, 448, 512, 512],
    "gradual_steep":  [128, 128, 128, 192, 256, 320, 384, 448, 512, 512],
}

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
    virtual_resize: bool,
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
    virtual_resize: if True, model always runs at 512px on a blurred (down→up) latent.
    """
    B = len(class_labels)
    prompt_data = encode_prompt(pipe, "dit", class_labels, device, dtype)

    latent_size = 512 // 8  # 64x64

    # Fixed noise vector, always kept at full 512-latent resolution
    eps = torch.randn(B, 4, latent_size, latent_size, device=device, dtype=dtype)

    # Bootstrap: noise a zero x0 to t=timesteps[0] to get the starting noisy latent
    t0 = timesteps[0]
    a0 = get_alpha(pipe, t0)
    latents = a0 * torch.zeros_like(eps) + (1 - a0 ** 2).sqrt() * eps

    for step_idx, (t, size) in enumerate(zip(timesteps, size_schedule)):
        lat_size = size // 8
        a_t = get_alpha(pipe, t)

        if virtual_resize:
            # Extract clean x0 estimate from the current noisy latent using the known eps,
            # blur it via downsample→upsample, then re-noise back to t before the forward pass.
            x0_est = (latents - (1 - a_t ** 2).sqrt() * eps) / a_t
            x0_blurred = upsample_latents(downsample_latents(x0_est, lat_size), latent_size)
            latents_input = a_t * x0_blurred + (1 - a_t ** 2).sqrt() * eps
            noise_pred = predict(pipe, "dit", latents_input, t, prompt_data, guidance_scale=CFG_SCALE)
            x0_full = (latents_input - (1 - a_t ** 2).sqrt() * noise_pred) / a_t
            x0_full = upsample_latents(downsample_latents(x0_full, lat_size), latent_size)
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

    pipe = load_model("dit", device, dtype)
    pipe.scheduler.set_timesteps(N_STEPS)
    timesteps = pipe.scheduler.timesteps

    print("DDIM timesteps:", timesteps.tolist())
    print(f"Running {len(SCHEDULES)} schedules x 2 modes = {len(SCHEDULES) * 2} conditions\n")

    if RANDOM_CLASSES:
        rng = torch.Generator().manual_seed(SEED)
        all_classes = torch.randint(0, 1000, (N_IMAGES,), generator=rng).tolist()
    else:
        # 10 images per class, 1000 classes = 10k images total
        all_classes = [c for c in range(1000) for _ in range(10)]

    generated_dirs: list[str] = []

    for sched_name, size_schedule in SCHEDULES.items():
        for virtual_resize in (False, True):
            mode = "virtual" if virtual_resize else "actual"
            cond_name = f"{sched_name}_{mode}"
            out_dir = os.path.join(OUT_DIR, cond_name)
            os.makedirs(out_dir, exist_ok=True)
            generated_dirs.append(out_dir)

            print(f"Generating {N_IMAGES} images [{cond_name}] schedule={size_schedule} -> {out_dir}")

            torch.manual_seed(SEED)  # same noise for all conditions
            img_idx = 0
            for start in tqdm(range(0, N_IMAGES, BATCH_SIZE), desc=cond_name):
                batch_classes = all_classes[start : start + BATCH_SIZE]
                images = generate_batch(pipe, batch_classes, size_schedule, timesteps, virtual_resize)
                for img in images:
                    img.save(os.path.join(out_dir, f"{img_idx:05d}.png"))
                    img_idx += 1

    # Bundle all generated images into a single zip
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    print(f"\nBundling images into {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as zf:
        for dir_path in generated_dirs:
            cond_name = os.path.basename(dir_path)
            for fname in sorted(os.listdir(dir_path)):
                if fname.endswith(".png"):
                    zf.write(
                        os.path.join(dir_path, fname),
                        arcname=os.path.join(cond_name, fname),
                    )

    print(f"Done. Zip written to {ZIP_PATH}")


if __name__ == "__main__":
    main()
