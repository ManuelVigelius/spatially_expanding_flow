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

Label-swap experiments (virtual resize only) are also run. In these runs the
class label is switched to the next sampled class at a chosen denoising step
(2nd, 4th, 6th, or 8th step = step indices 1, 3, 5, 7). The "next" label is
the next entry in the sampled all_classes list (with wraparound).

Classes are sampled uniformly from [0, 999] (ImageNet-1k), 10 per class.
CFG scale is 4.0. All images saved as PNGs; results bundled into a zip.

Images are saved as PNGs under:
  results/fid_images/<schedule_name>_<mode>/
A final zip is written to:
  results/fid_images.zip
A metadata JSON is written to:
  results/fid_images_metadata.json
"""

import json
import os
import urllib.request
import zipfile

import torch
from PIL import Image
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32

from models import encode_prompt, load_model, predict, downsample_latents, upsample_latents

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_IMAGES = 32
BATCH_SIZE = 32
N_STEPS = 10
CFG_SCALE = 6.0
OUT_DIR = "results/fid_images"
ZIP_PATH = "results/fid_images.zip"
METADATA_PATH = "results/fid_images_metadata.json"
SEED = 0

# Step indices (0-based) at which to swap the label in label-swap experiments.
# Step 0 is the noisiest step. Steps 1,3,5,7 correspond to the 2nd,4th,6th,8th steps.
LABEL_SWAP_STEPS = [1, 3, 5, 7]

# URL for human-readable ImageNet class names (1000 entries, index == class id)
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)

# If True, sample class labels uniformly at random (seeded by SEED).
# If False, use deterministic labels: 10 images per class, all 1000 classes.
RANDOM_CLASSES = True

# ---------------------------------------------------------------------------
# Hand-written schedules (pixel sizes, one per denoising step, noisiest first)
# ---------------------------------------------------------------------------
SCHEDULES: dict[str, list[int]] = {
    "full":           [512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
    "early_small":    [256, 256, 256, 256, 256, 512, 512, 512, 512, 512],
    # "gradual_mild":   [256, 256, 320, 320, 384, 384, 448, 448, 512, 512],
    # "gradual_medium": [192, 192, 256, 256, 320, 384, 448, 448, 512, 512],
    "gradual_steep":  [128, 128, 128, 192, 256, 320, 384, 448, 512, 512],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_imagenet_labels() -> list[str]:
    """Download and return the 1000 ImageNet class name strings (index == class id)."""
    with urllib.request.urlopen(IMAGENET_LABELS_URL) as resp:
        return json.loads(resp.read())


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


@torch.no_grad()
def generate_batch_label_swap(
    pipe,
    class_labels: list[int],
    next_class_labels: list[int],
    size_schedule: list[int],
    timesteps: torch.Tensor,
    swap_step: int,
) -> list[Image.Image]:
    """
    Like generate_batch with virtual_resize=True, but the class label is switched
    from class_labels to next_class_labels at step index `swap_step`.

    next_class_labels: the label used from swap_step onward (one per image in the batch).
    swap_step: 0-based step index at which the swap occurs.
    """
    B = len(class_labels)
    latent_size = 512 // 8  # 64x64

    prompt_data_before = encode_prompt(pipe, "dit", class_labels, device, dtype)
    prompt_data_after = encode_prompt(pipe, "dit", next_class_labels, device, dtype)

    eps = torch.randn(B, 4, latent_size, latent_size, device=device, dtype=dtype)

    t0 = timesteps[0]
    a0 = get_alpha(pipe, t0)
    latents = a0 * torch.zeros_like(eps) + (1 - a0 ** 2).sqrt() * eps

    for step_idx, (t, size) in enumerate(zip(timesteps, size_schedule)):
        lat_size = size // 8
        a_t = get_alpha(pipe, t)
        prompt_data = prompt_data_after if step_idx >= swap_step else prompt_data_before

        x0_est = (latents - (1 - a_t ** 2).sqrt() * eps) / a_t
        x0_blurred = upsample_latents(downsample_latents(x0_est, lat_size), latent_size)
        latents_input = a_t * x0_blurred + (1 - a_t ** 2).sqrt() * eps
        noise_pred = predict(pipe, "dit", latents_input, t, prompt_data, guidance_scale=CFG_SCALE)
        x0_full = (latents_input - (1 - a_t ** 2).sqrt() * noise_pred) / a_t
        x0_full = upsample_latents(downsample_latents(x0_full, lat_size), latent_size)

        if step_idx + 1 < len(timesteps):
            t_prev = timesteps[step_idx + 1]
            a_prev = get_alpha(pipe, t_prev)
        else:
            a_prev = torch.ones(1, device=device, dtype=dtype).view(1, 1, 1, 1)

        latents = a_prev * x0_full + (1 - a_prev ** 2).sqrt() * eps

    return decode_latents(pipe, latents)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)

    print("Fetching ImageNet class names ...")
    imagenet_labels = fetch_imagenet_labels()

    pipe = load_model("dit", device, dtype)
    pipe.scheduler.set_timesteps(N_STEPS)
    timesteps = pipe.scheduler.timesteps

    print("DDIM timesteps:", timesteps.tolist())

    n_swap_conds = len(SCHEDULES) * len(LABEL_SWAP_STEPS)
    print(
        f"Running {len(SCHEDULES)} schedules x 2 modes = {len(SCHEDULES) * 2} conditions"
        f" + {n_swap_conds} label-swap conditions\n"
    )

    if RANDOM_CLASSES:
        rng = torch.Generator().manual_seed(SEED)
        all_classes = torch.randint(0, 1000, (N_IMAGES,), generator=rng).tolist()
    else:
        # 10 images per class, 1000 classes = 10k images total
        all_classes = [c for c in range(1000) for _ in range(10)]

    # next_classes[i] is the label used after swapping for image i (next entry, with wraparound)
    next_classes = [all_classes[(i + 1) % len(all_classes)] for i in range(len(all_classes))]

    generated_dirs: list[str] = []
    # metadata maps "cond_name/00000.png" -> dict with generation info
    metadata: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Standard conditions (actual + virtual resize, no label swap)
    # ------------------------------------------------------------------
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
                for img, cls in zip(images, batch_classes):
                    fname = f"{img_idx:05d}.png"
                    img.save(os.path.join(out_dir, fname))
                    metadata[f"{cond_name}/{fname}"] = {
                        "condition": cond_name,
                        "schedule": sched_name,
                        "size_schedule": size_schedule,
                        "mode": mode,
                        "class_id": cls,
                        "class_name": imagenet_labels[cls],
                        "label_swap_step": None,
                        "swap_class_id": None,
                        "swap_class_name": None,
                    }
                    img_idx += 1

    # ------------------------------------------------------------------
    # Label-swap conditions (virtual resize only)
    # ------------------------------------------------------------------
    for sched_name, size_schedule in SCHEDULES.items():
        for swap_step in LABEL_SWAP_STEPS:
            cond_name = f"{sched_name}_virtual_swap_at_step{swap_step}"
            out_dir = os.path.join(OUT_DIR, cond_name)
            os.makedirs(out_dir, exist_ok=True)
            generated_dirs.append(out_dir)

            print(
                f"Generating {N_IMAGES} images [{cond_name}]"
                f" schedule={size_schedule} swap_step={swap_step} -> {out_dir}"
            )

            torch.manual_seed(SEED)  # same noise for all conditions
            img_idx = 0
            for start in tqdm(range(0, N_IMAGES, BATCH_SIZE), desc=cond_name):
                batch_classes = all_classes[start : start + BATCH_SIZE]
                batch_next = next_classes[start : start + BATCH_SIZE]
                images = generate_batch_label_swap(
                    pipe, batch_classes, batch_next, size_schedule, timesteps, swap_step
                )
                for img, cls, next_cls in zip(images, batch_classes, batch_next):
                    fname = f"{img_idx:05d}.png"
                    img.save(os.path.join(out_dir, fname))
                    metadata[f"{cond_name}/{fname}"] = {
                        "condition": cond_name,
                        "schedule": sched_name,
                        "size_schedule": size_schedule,
                        "mode": "virtual",
                        "class_id": cls,
                        "class_name": imagenet_labels[cls],
                        "label_swap_step": swap_step,
                        "swap_class_id": next_cls,
                        "swap_class_name": imagenet_labels[next_cls],
                    }
                    img_idx += 1

    # ------------------------------------------------------------------
    # Write metadata JSON
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {METADATA_PATH}")

    # Bundle all generated images into a single zip
    print(f"Bundling images into {ZIP_PATH} ...")
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
