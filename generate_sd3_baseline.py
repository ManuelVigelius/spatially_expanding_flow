"""
Generate images with SD3 in two conditions:
  - baseline:       standard pipeline, no modifications
  - virtual_resize: manual denoising loop with virtual resize at each step
                    (downsample→upsample the latent to blur high-freq content
                    before the forward pass, model always runs at full resolution)

Both conditions use 10 steps and the same seed/prompt for a fair comparison.
"""

import os
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from tqdm import tqdm

from config import device, dtype
from models import encode_prompt, predict, downsample_latents, upsample_latents

PROMPT = "A photo of a cat sitting on a windowsill"
N_IMAGES = 4
N_STEPS = 10
GUIDANCE_SCALE = 7.0
# Virtual resize: pixel size to blur down to at each step (single fixed size for simplicity)
VIRTUAL_SIZE = 256  # pixels → latent size = 256 // 8 = 32
OUT_DIR = "results/sd3_baseline"
SEED = 42


def decode_latents(pipe, latents: torch.Tensor) -> list[Image.Image]:
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    images = pipe.vae.decode(latents, return_dict=False)[0]  # [B, 3, H, W] in [-1, 1]
    images = (images.clamp(-1, 1) + 1) / 2
    images = (images * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(img) for img in images]


@torch.no_grad()
def generate_virtual_resize(pipe, prompt_data, B: int) -> list[Image.Image]:
    """Manual SD3 denoising loop with virtual resize at every step."""
    lat_size = 512 // 8  # 64
    virt_lat_size = VIRTUAL_SIZE // 8  # 32

    # SD3 starts from pure noise
    torch.manual_seed(SEED)
    noise = torch.randn(B, 16, lat_size, lat_size, device=device, dtype=dtype)
    latents = noise.clone()

    sigmas = pipe.scheduler.sigmas  # shape [N_STEPS+1], descending 1→0
    timesteps = pipe.scheduler.timesteps  # shape [N_STEPS]

    for i, (t, sigma) in enumerate(tqdm(zip(timesteps, sigmas[:-1]), total=N_STEPS, desc="virtual_resize")):
        sigma = sigma.to(device=device, dtype=dtype).view(1, 1, 1, 1)

        # Estimate x0 from current latent: x_t = (1-σ)*x0 + σ*noise  →  x0 = (x_t - σ*noise) / (1-σ)
        # Equivalently using velocity: x0 = x_t - σ * v  (we get v from the model below)
        # Virtual resize: blur x0 estimate via down→up, re-noise back to current sigma
        x0_est = (latents - sigma * noise) / (1 - sigma).clamp(min=1e-6)
        x0_blurred = upsample_latents(downsample_latents(x0_est, virt_lat_size), lat_size)
        latents_input = (1 - sigma) * x0_blurred + sigma * noise

        vel = predict(pipe, "sd3", latents_input, t, prompt_data, guidance_scale=GUIDANCE_SCALE)

        # x0 from blurred latent + model velocity
        x0_full = latents_input - sigma * vel
        x0_full = upsample_latents(downsample_latents(x0_full, virt_lat_size), lat_size)

        # Re-noise to next sigma using the same fixed noise
        sigma_next = sigmas[i + 1].to(device=device, dtype=dtype).view(1, 1, 1, 1)
        latents = (1 - sigma_next) * x0_full + sigma_next * noise

    return decode_latents(pipe, latents)


def main():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=dtype,
    ).to(device)
    pipe.scheduler.set_timesteps(N_STEPS, device=device)

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Baseline: standard pipeline ---
    print("Generating baseline images...")
    generator = torch.Generator(device=device).manual_seed(SEED)
    baseline_images = pipe(
        prompt=PROMPT,
        num_images_per_prompt=N_IMAGES,
        num_inference_steps=N_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images
    for i, img in enumerate(baseline_images):
        path = os.path.join(OUT_DIR, f"baseline_{i:04d}.png")
        img.save(path)
        print(f"  Saved {path}")

    # --- Virtual resize condition ---
    print(f"\nGenerating virtual_resize images (blur to {VIRTUAL_SIZE}px at each step)...")
    prompt_data = encode_prompt(pipe, "sd3", PROMPT, device, dtype)
    vr_images = generate_virtual_resize(pipe, prompt_data, N_IMAGES)
    for i, img in enumerate(vr_images):
        path = os.path.join(OUT_DIR, f"virtual_resize_{i:04d}.png")
        img.save(path)
        print(f"  Saved {path}")

    print(f"\nDone. Images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
