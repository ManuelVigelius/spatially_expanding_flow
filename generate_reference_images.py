"""
Generate SDXL reference latents (512x512) and save them to Google Drive in bfloat16.

For each image we run denoising steps with SDXL (unconditional, no CFG) starting from
reproducible noise (see noise_generator.py). Both the initial noise latent and the final
denoised latent are saved as .pt files in bfloat16.

Usage:
    python generate_reference_images.py \
        --num_images 1000 \
        --batch_size 4 \
        --output_dir /path/to/gdrive/reference_latents
"""

import argparse
import os

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm import tqdm

import config
from noise_generator import get_img_noise


NUM_INFERENCE_STEPS = 28

# SDXL latent shape for 512x512: VAE downsamples by 8, 4 channels
_LATENT_H = 64
_LATENT_W = 64
_LATENT_C = 4


def run_sdxl_denoising(pipe, latents, prompt_embeds, pooled_embeds, add_time_ids):
    """Run unconditional denoising steps (guidance_scale=1) and return the final latents."""
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=config.device)

    # Scale initial noise by scheduler's sigma
    latents = latents * pipe.scheduler.init_noise_sigma

    for t in tqdm(pipe.scheduler.timesteps, desc="Denoising", leave=False):
        with torch.no_grad():
            latents_input = pipe.scheduler.scale_model_input(latents, t)
            velocity = pipe.unet(
                latents_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
                return_dict=False,
            )[0]
        latents = pipe.scheduler.step(velocity, t, latents, return_dict=False)[0]

    return latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to Google Drive folder for saving latents")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Global start index for noise seeds (for resuming)")
    parser.add_argument("--test", action="store_true",
                        help="Run a single batch only and also save decoded PNGs")
    args = parser.parse_args()

    if args.test:
        args.num_images = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=config.dtype,
    ).to(config.device)
    pipe.unet.eval()

    with torch.no_grad():
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            prompt="",
            prompt_2="",
            device=config.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    # add_time_ids encodes (original_size, crop_coords, target_size) for SDXL
    add_time_ids = pipe._get_add_time_ids(
        original_size=(config.height, config.width),
        crops_coords_top_left=(0, 0),
        target_size=(config.height, config.width),
        dtype=config.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    ).to(config.device)

    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        global_start = args.start_index + batch_idx * args.batch_size
        current_batch = min(args.batch_size, args.num_images - batch_idx * args.batch_size)

        noise = get_img_noise(
            current_batch, start_index=global_start,
            device=config.device, latent_c=_LATENT_C,
        )
        noise = noise.to(dtype=config.dtype)

        batch_prompt_embeds = prompt_embeds.expand(current_batch, -1, -1)
        batch_pooled_embeds = pooled_embeds.expand(current_batch, -1)
        batch_time_ids = add_time_ids.expand(current_batch, -1)

        final_latents = run_sdxl_denoising(
            pipe, noise,
            batch_prompt_embeds, batch_pooled_embeds, batch_time_ids,
        )

        if args.test:
            with torch.no_grad():
                decoded = pipe.vae.decode(
                    final_latents / pipe.vae.config.scaling_factor,
                    return_dict=False,
                )[0]
            decoded = ((decoded.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu()

        for j in range(current_batch):
            img_idx = global_start + j
            torch.save(noise[j].to(torch.bfloat16).cpu(),
                       os.path.join(args.output_dir, f"noise_{img_idx:06d}.pt"))
            torch.save(final_latents[j].to(torch.bfloat16).cpu(),
                       os.path.join(args.output_dir, f"latent_{img_idx:06d}.pt"))

            if args.test:
                img = Image.fromarray(decoded[j].permute(1, 2, 0).numpy())
                img.save(os.path.join(args.output_dir, f"image_{img_idx:06d}.png"))

    print(f"Done. {args.num_images} latent pairs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
