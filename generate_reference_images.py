"""
Generate DiT-XL-2 reference images (512x512) and save latents to disk in bfloat16.

Unconditional generation: class label 1000 (null class) with guidance_scale=1.

For each image we run denoising with DiT-XL-2 starting from reproducible noise
(see noise_generator.py). Both the initial noise latent and the final denoised
latent are saved as .pt files in bfloat16.

Usage:
    python generate_reference_images.py \
        --num_images 1000 \
        --batch_size 4 \
        --output_dir /path/to/output
"""

import argparse
import os

import torch
from diffusers import DiTPipeline
from PIL import Image
from tqdm import tqdm

import config
from noise_generator import get_class_labels, get_img_noise


NUM_INFERENCE_STEPS = 100

_LATENT_H = 64  # 512 // 8
_LATENT_W = 64
_LATENT_C = 4

# Class label 1000 is the null/unconditional label in DiT
_NULL_CLASS = 1000


def run_dit_denoising(pipe, latents, class_labels, guidance_scale=1.0):
    """Run denoising and return the final latents.

    Args:
        class_labels: Long tensor of shape [batch_size] with ImageNet class indices.
        guidance_scale: CFG scale. 1.0 disables guidance.
    """
    batch_size = latents.shape[0]
    do_cfg = guidance_scale > 1.0
    cond_labels = class_labels
    uncond_labels = torch.tensor([_NULL_CLASS] * batch_size, device=config.device)

    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=config.device)
    latents = latents * pipe.scheduler.init_noise_sigma

    for t in tqdm(pipe.scheduler.timesteps, desc="Denoising", leave=False):
        with torch.no_grad():
            latents_input = pipe.scheduler.scale_model_input(latents, t)
            timesteps = t.unsqueeze(0).expand(batch_size)

            if do_cfg:
                combined = torch.cat([latents_input, latents_input])
                combined_labels = torch.cat([uncond_labels, cond_labels])
                combined_timesteps = timesteps.repeat(2)
                out = pipe.transformer(
                    combined, timestep=combined_timesteps,
                    class_labels=combined_labels, return_dict=False,
                )[0]
                uncond_out, cond_out = out.chunk(2, dim=0)
                # split off variance prediction
                uncond_noise, _ = uncond_out.chunk(2, dim=1)
                cond_noise, _ = cond_out.chunk(2, dim=1)
                noise_pred = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
            else:
                noise_pred = pipe.transformer(
                    latents_input, timestep=timesteps,
                    class_labels=cond_labels, return_dict=False,
                )[0]
                noise_pred, _ = noise_pred.chunk(2, dim=1)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_index", type=int, default=0,
                        help="Global start index for noise seeds (for resuming)")
    parser.add_argument("--test", action="store_true",
                        help="Run a single batch only and also save decoded PNGs")
    parser.add_argument("--test-volcano", action="store_true",
                        help="Like --test but class 980 (volcano) with CFG scale 4.0")
    args = parser.parse_args()

    if args.test or args.test_volcano:
        args.num_images = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = DiTPipeline.from_pretrained(
        "facebook/DiT-XL-2-512",
        torch_dtype=config.dtype,
    ).to(config.device)
    pipe.transformer.eval()

    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        global_start = args.start_index + batch_idx * args.batch_size
        current_batch = min(args.batch_size, args.num_images - batch_idx * args.batch_size)

        noise = get_img_noise(
            current_batch, start_index=global_start,
            device=config.device, latent_c=_LATENT_C,
        ).to(dtype=config.dtype)

        if args.test_volcano:
            class_labels = torch.tensor([980] * current_batch)
        else:
            class_labels = get_class_labels(current_batch, start_index=global_start)
        class_labels = class_labels.to(config.device)

        final_latents = run_dit_denoising(pipe, noise, class_labels=class_labels, guidance_scale=4.0)

        if args.test or args.test_volcano:
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
            torch.save(class_labels[j].cpu(),
                       os.path.join(args.output_dir, f"class_{img_idx:06d}.pt"))

            if args.test or args.test_volcano:
                img = Image.fromarray(decoded[j].permute(1, 2, 0).numpy())
                img.save(os.path.join(args.output_dir, f"image_{img_idx:06d}.png"))

    print(f"Done. {args.num_images} latent pairs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
