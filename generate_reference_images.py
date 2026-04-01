"""
Generate SD3 reference latents (512x512) and save them to Google Drive in bfloat16.

For each image we run 10 denoising steps with SD3 starting from reproducible noise
(see noise_generator.py). Both the initial noise latent and the final denoised latent
are saved as .pt files in bfloat16.

Usage:
    python generate_reference_images.py \
        --num_images 1000 \
        --batch_size 4 \
        --output_dir /path/to/gdrive/reference_latents \
        --prompt ""
"""

import argparse
import os

import torch
from diffusers import StableDiffusion3Pipeline

import config
from noise_generator import get_img_noise


NUM_INFERENCE_STEPS = 10


def prepare_latents(pipe, noise: torch.Tensor) -> torch.Tensor:
    """Scale raw noise into the SD3 latent space (shift + scale factor applied)."""
    # SD3 pipeline scales the initial noise by the scheduler's init_noise_sigma
    noise = noise * pipe.scheduler.init_noise_sigma
    return noise


def run_sd3_denoising(pipe, latents: torch.Tensor, prompt_embeds, pooled_embeds) -> torch.Tensor:
    """Run NUM_INFERENCE_STEPS of SD3 denoising and return the final latents."""
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=config.device)
    timesteps = pipe.scheduler.timesteps

    for t in timesteps:
        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=t.unsqueeze(0).to(config.device),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


def decode_and_unnormalize(pipe, latents: torch.Tensor) -> torch.Tensor:
    """Invert the VAE scaling and return latents ready for vae.decode (if needed)."""
    return latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to Google Drive folder for saving latents")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for all images (empty = unconditional)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Global start index for noise seeds (for resuming)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=config.dtype,
    ).to(config.device)
    pipe.transformer.eval()

    with torch.no_grad():
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            prompt=args.prompt,
            prompt_2=args.prompt,
            prompt_3=args.prompt,
            device=config.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        global_start = args.start_index + batch_idx * args.batch_size
        current_batch = min(args.batch_size, args.num_images - batch_idx * args.batch_size)

        # Generate noise sequentially so each image always gets the same noise
        noise = get_img_noise(current_batch, start_index=global_start, device=config.device)
        noise = noise.to(dtype=config.dtype)

        latents = prepare_latents(pipe, noise)

        # Expand prompt embeddings to match batch size
        batch_prompt_embeds = prompt_embeds.expand(current_batch, -1, -1)
        batch_pooled_embeds = pooled_embeds.expand(current_batch, -1)

        final_latents = run_sd3_denoising(pipe, latents, batch_prompt_embeds, batch_pooled_embeds)

        # Save each image individually in bfloat16
        for j in range(current_batch):
            img_idx = global_start + j
            noise_bf16 = noise[j].to(torch.bfloat16).cpu()
            latent_bf16 = final_latents[j].to(torch.bfloat16).cpu()

            torch.save(noise_bf16, os.path.join(args.output_dir, f"noise_{img_idx:06d}.pt"))
            torch.save(latent_bf16, os.path.join(args.output_dir, f"latent_{img_idx:06d}.pt"))

        print(f"Saved images {global_start} - {global_start + current_batch - 1}")

    print(f"Done. {args.num_images} latent pairs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
