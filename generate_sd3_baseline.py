"""
Generate images with SD3 in multiple conditions:
  - baseline:       standard pipeline call at 512px
  - manual:         our own Euler loop at 512px — should be identical to baseline
  - virtual_resize: manual loop with per-step latent blurring (down→up before forward pass)

Per-step pixel size schedules (noisiest → cleanest), matching generate_fid_images.py:
  full           — all steps at 512px (no-op, should match manual)
  early_small    — 256px for first 5 noisy steps, then 512px
  gradual_mild   — linearly ramp from 256px up to 512px
  gradual_medium — steeper ramp from 192px up to 512px
  gradual_steep  — aggressive ramp from 128px up to 512px
"""

import os
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F


PROMPT = "A photo of a cat sitting on a windowsill"
N_IMAGES = 4
N_STEPS = 10
GUIDANCE_SCALE = 7.0
IMAGE_SIZE = 1024
OUT_DIR = "results/sd3_baseline"
SEED = 42

SCHEDULES: dict[str, list[int]] = {
    "full":           [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
    "early_small":    [ 512,  512,  512,  512,  512, 1024, 1024, 1024, 1024, 1024],
    "gradual_mild":   [ 512,  512,  640,  640,  768,  768,  896,  896, 1024, 1024],
    "gradual_medium": [ 384,  384,  512,  512,  640,  768,  896,  896, 1024, 1024],
    "gradual_steep":  [ 256,  256,  256,  384,  512,  640,  768,  896, 1024, 1024],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipeline(device, dtype):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=dtype,
    )
    return pipe.to(device)


def encode_text(pipe, prompt: str, device):
    """Encode text prompt (and empty string for CFG) into embeddings."""
    with torch.no_grad():
        embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        uncond_embeds, _, uncond_pooled, _ = pipe.encode_prompt(
            prompt="", prompt_2="", prompt_3="",
            device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return {
        "embeds": embeds,
        "pooled": pooled,
        "uncond_embeds": uncond_embeds,
        "uncond_pooled": uncond_pooled,
    }


def transformer_forward(pipe, latents, timestep, prompt_data, guidance_scale, device):
    """Run SD3 transformer with CFG and return velocity [B, C, H, W]."""
    B = latents.shape[0]
    embeds = prompt_data["embeds"].expand(B, -1, -1)
    pooled = prompt_data["pooled"].expand(B, -1)
    uncond_embeds = prompt_data["uncond_embeds"].expand(B, -1, -1)
    uncond_pooled = prompt_data["uncond_pooled"].expand(B, -1)

    embeds_in = torch.cat([uncond_embeds, embeds])
    pooled_in = torch.cat([uncond_pooled, pooled])
    latents_in = latents.repeat(2, 1, 1, 1)

    t = timestep.unsqueeze(0).to(device) if timestep.dim() == 0 else timestep.to(device)
    with torch.no_grad():
        out = pipe.transformer(
            hidden_states=latents_in,
            timestep=t,
            encoder_hidden_states=embeds_in,
            pooled_projections=pooled_in,
            return_dict=False,
        )[0]
    vel_uncond, vel_cond = out.chunk(2)
    return vel_uncond + guidance_scale * (vel_cond - vel_uncond)


def decode_latents(pipe, latents: torch.Tensor) -> list[Image.Image]:
    """VAE-decode SD3 latents [B, C, H, W] to a list of PIL images."""
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.no_grad():
        images = pipe.vae.decode(latents, return_dict=False)[0]  # [B, 3, H, W] in [-1, 1]
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(img) for img in images]


def downsample(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[2] == size and x.shape[3] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=True)


def upsample(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[2] == size and x.shape[3] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=True)


def save_grid(images: list[Image.Image], path: str, cols: int = 4) -> None:
    """Save a list of PIL images as a grid."""
    n = len(images)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (w * cols, h * rows))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    grid.save(path)


# ---------------------------------------------------------------------------
# Generation modes
# ---------------------------------------------------------------------------

def generate_baseline(pipe, n_images: int, device) -> list[Image.Image]:
    """Standard pipeline call — the simplest possible baseline."""
    torch.manual_seed(SEED)
    result = pipe(
        prompt=PROMPT,
        num_images_per_prompt=n_images,
        num_inference_steps=N_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        generator=torch.Generator(device=device).manual_seed(SEED),
    )
    return result.images


@torch.no_grad()
def generate_manual(pipe, prompt_data, n_images: int, device, dtype) -> list[Image.Image]:
    """
    Hand-rolled Euler flow-matching loop at full 512px resolution.
    Should produce results visually indistinguishable from baseline.

    Note: a small numerical difference from the pipeline is expected because
    SD3's FlowMatchEulerDiscreteScheduler applies a learned shift (mu) to the
    sigmas when called from the pipeline (based on image resolution), whereas
    our manual set_timesteps call uses the default shift. The images will look
    the same but won't be pixel-identical.
    """
    torch.manual_seed(SEED)
    lat_size = IMAGE_SIZE // 8  # 64

    pipe.scheduler.set_timesteps(N_STEPS, device=device)
    sigmas = pipe.scheduler.sigmas  # length N_STEPS + 1, descending (noisy→clean)
    timesteps = pipe.scheduler.timesteps  # length N_STEPS

    # Initial noise at t=1 (pure noise)
    latents = torch.randn(n_images, 16, lat_size, lat_size, device=device, dtype=dtype)
    latents = latents * sigmas[0]

    for step_idx, (t, sigma) in enumerate(zip(timesteps, sigmas[:-1])):
        sigma_next = sigmas[step_idx + 1]
        vel = transformer_forward(pipe, latents, t, prompt_data, GUIDANCE_SCALE, device)
        # Euler step: x_{t-1} = x_t + (sigma_next - sigma) * vel
        latents = latents + (sigma_next - sigma) * vel

    return decode_latents(pipe, latents)


@torch.no_grad()
def generate_virtual_resize(
    pipe,
    prompt_data,
    n_images: int,
    size_schedule: list[int],
    device,
    dtype,
) -> list[Image.Image]:
    """
    Euler loop where at each step we:
      1. Run the transformer on the current latent to get velocity.
      2. Extract x0 from the velocity prediction: x0 = x_t - sigma * vel.
      3. Blur x0 via downsample→upsample to the step's target resolution.
      4. Re-noise the blurred x0 with the fixed noise eps to sigma_next.

    The fixed noise vector eps is used only for re-noising (not to reconstruct
    x0 from the input latent, which would break the Euler trajectory).

    When size_schedule is all-512 ("full"), no blurring occurs (downsample then
    upsample to the same size is a no-op), so the loop is identical to the
    plain manual Euler loop.
    """
    torch.manual_seed(SEED)
    lat_size = IMAGE_SIZE // 8  # 64

    pipe.scheduler.set_timesteps(N_STEPS, device=device)
    sigmas = pipe.scheduler.sigmas
    timesteps = pipe.scheduler.timesteps

    # Start from pure noise (same as manual loop)
    latents = sigmas[0] * torch.randn(n_images, 16, lat_size, lat_size, device=device, dtype=dtype)

    for step_idx, (t, sigma) in enumerate(zip(timesteps, sigmas[:-1])):
        sigma_next = sigmas[step_idx + 1]
        target_px = size_schedule[step_idx]
        target_lat = target_px // 8

        # Forward pass on the current (unmodified) latent
        vel = transformer_forward(pipe, latents, t, prompt_data, GUIDANCE_SCALE, device)

        # vel = eps - x0, so:
        #   x0  = x_t - sigma * vel
        #   eps = x_t + (1 - sigma) * vel
        x0_pred  = latents - sigma * vel
        eps_pred = latents + (1 - sigma) * vel

        # Blur x0 (no-op when target_lat == lat_size)
        x0_pred_blurred = upsample(downsample(x0_pred, target_lat), lat_size)

        # Re-noise blurred x0 with the predicted noise to sigma_next
        latents = (1 - sigma_next) * x0_pred_blurred + sigma_next * eps_pred

    return decode_latents(pipe, latents)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from config import device, dtype

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading SD3 pipeline...")
    pipe = load_pipeline(device, dtype)

    print("Encoding text prompt...")
    prompt_data = encode_text(pipe, PROMPT, device)

    # ------------------------------------------------------------------
    # 1. Baseline: standard pipeline
    # ------------------------------------------------------------------
    print("\n[1/3] Generating baseline images (standard pipeline)...")
    torch.manual_seed(SEED)
    baseline_images = generate_baseline(pipe, N_IMAGES, device)
    baseline_dir = os.path.join(OUT_DIR, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    for i, img in enumerate(baseline_images):
        img.save(os.path.join(baseline_dir, f"{i:04d}.png"))
    save_grid(baseline_images, os.path.join(OUT_DIR, "baseline_grid.png"))
    print(f"  Saved {len(baseline_images)} images to {baseline_dir}/")

    # ------------------------------------------------------------------
    # 2. Manual loop at full resolution (sanity check vs. baseline)
    # ------------------------------------------------------------------
    print("\n[2/3] Generating manual-loop images (should match baseline)...")
    manual_images = generate_manual(pipe, prompt_data, N_IMAGES, device, dtype)
    manual_dir = os.path.join(OUT_DIR, "manual")
    os.makedirs(manual_dir, exist_ok=True)
    for i, img in enumerate(manual_images):
        img.save(os.path.join(manual_dir, f"{i:04d}.png"))
    save_grid(manual_images, os.path.join(OUT_DIR, "manual_grid.png"))
    print(f"  Saved {len(manual_images)} images to {manual_dir}/")

    # ------------------------------------------------------------------
    # 3. Virtual resize with each schedule
    # ------------------------------------------------------------------
    print(f"\n[3/3] Generating virtual-resize images ({len(SCHEDULES)} schedules)...")
    for sched_name, size_schedule in tqdm(SCHEDULES.items(), desc="schedules"):
        images = generate_virtual_resize(pipe, prompt_data, N_IMAGES, size_schedule, device, dtype)
        sched_dir = os.path.join(OUT_DIR, f"virtual_{sched_name}")
        os.makedirs(sched_dir, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(sched_dir, f"{i:04d}.png"))
        save_grid(images, os.path.join(OUT_DIR, f"virtual_{sched_name}_grid.png"))
        print(f"  [{sched_name}] schedule={size_schedule}")

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
