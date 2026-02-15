import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models import load_model, encode_prompt_for_model, apply_spatial_compression


def _get_caption(sample, use_caption):
    """Extract caption from a dataset sample."""
    if use_caption and "captions" in sample and sample["captions"]:
        captions = sample["captions"]
        return captions[0] if isinstance(captions, list) else captions
    return ""


def _preprocess_image(sample, width, height, device, dtype):
    """Convert a dataset sample image to a normalized tensor."""
    image = sample["image"].convert("RGB")
    image = image.resize((width, height))
    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device=device, dtype=dtype)
    return (image / 127.5) - 1.0


def run_experiment_sd3(pipe, dataset_samples, use_caption, config):
    """Run experiment with SD3 model."""
    pipe.scheduler.set_timesteps(config.num_inference_steps)
    pipe.transformer.eval()

    results = {t.item(): {**{size: [] for size in config.compression_sizes}, 'baseline': []} for t in config.t_indices}

    for sample in tqdm(dataset_samples, desc=f"SD3 (caption={use_caption})"):
        with torch.no_grad():
            prompt = _get_caption(sample, use_caption)
            prompt_embeds, pooled_prompt_embeds = encode_prompt_for_model(pipe, "sd3", prompt, config.device, config.dtype)

            image = _preprocess_image(sample, config.width, config.height, config.device, config.dtype)

            latents = pipe.vae.encode(image).latent_dist.sample()
            latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)

            for t_idx in config.t_indices:
                t_idx_int = t_idx.item()
                sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=config.dtype)

                noisy_latents = (1 - sigma) * latents + sigma * noise
                velocity_target = noise - latents

                timestep = pipe.scheduler.timesteps[t_idx_int].unsqueeze(0).to(config.device)

                velocity_pred = pipe.transformer(
                    hidden_states=noisy_latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False
                )[0]

                baseline_loss = F.mse_loss(velocity_pred, velocity_target)
                results[t_idx_int]['baseline'].append(baseline_loss.item())

                latent_pred = noise - velocity_pred

                for compressed_size in config.compression_sizes:
                    compressed_latent_pred = apply_spatial_compression(latent_pred, compressed_size)
                    reconstructed_velocity_pred = noise - compressed_latent_pred
                    loss = F.mse_loss(reconstructed_velocity_pred, velocity_target)
                    results[t_idx_int][compressed_size].append(loss.item())

    return results


def run_experiment_flux(pipe, dataset_samples, use_caption, config):
    """Run experiment with FLUX model."""
    pipe.transformer.eval()

    results = {t.item(): {**{size: [] for size in config.compression_sizes}, 'baseline': []} for t in config.t_indices}

    latent_height = config.height // 8
    latent_width = config.width // 8
    packed_height = latent_height // 2
    packed_width = latent_width // 2

    image_seq_len = packed_height * packed_width
    base_seq_len = pipe.scheduler.config.base_image_seq_len
    max_seq_len = pipe.scheduler.config.max_image_seq_len
    mu = pipe.scheduler.config.base_shift + pipe.scheduler.config.max_shift * (
        (image_seq_len - base_seq_len) / (max_seq_len - base_seq_len)
    )
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=config.device, mu=mu)

    guidance_scale = 3.5
    guidance = torch.tensor([guidance_scale], device=config.device, dtype=config.dtype)

    latent_image_ids = pipe._prepare_latent_image_ids(1, packed_height, packed_width, config.device, config.dtype)

    for sample in tqdm(dataset_samples, desc=f"FLUX (caption={use_caption})"):
        with torch.no_grad():
            prompt = _get_caption(sample, use_caption)
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_for_model(pipe, "flux", prompt, config.device, config.dtype)

            image = _preprocess_image(sample, config.width, config.height, config.device, config.dtype)

            latents = pipe.vae.encode(image).latent_dist.sample()
            latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

            packed_latents = pipe._pack_latents(latents, 1, pipe.transformer.config.in_channels // 4, latent_height, latent_width)

            noise = torch.randn_like(packed_latents)

            for t_idx in config.t_indices:
                t_idx_int = t_idx.item()
                sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=config.dtype)

                noisy_latents = (1 - sigma) * packed_latents + sigma * noise
                velocity_target = noise - packed_latents

                timestep = pipe.scheduler.timesteps[t_idx_int].expand(1).to(config.device) / 1000.0

                velocity_pred = pipe.transformer(
                    hidden_states=noisy_latents,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False
                )[0]

                baseline_loss = F.mse_loss(velocity_pred, velocity_target)
                results[t_idx_int]['baseline'].append(baseline_loss.item())

                latent_pred = noise - velocity_pred

                for compressed_size in config.compression_sizes:
                    batch_size, num_patches, channels = latent_pred.shape
                    latent_h = config.height // pipe.vae_scale_factor
                    latent_w = config.width // pipe.vae_scale_factor
                    num_channels = channels // 4

                    unpacked = latent_pred.view(batch_size, latent_h // 2, latent_w // 2, num_channels, 2, 2)
                    unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
                    unpacked_latent_pred = unpacked.reshape(batch_size, num_channels, latent_h, latent_w)

                    compressed_unpacked = apply_spatial_compression(unpacked_latent_pred, compressed_size)

                    packed = compressed_unpacked.view(batch_size, num_channels, latent_h // 2, 2, latent_w // 2, 2)
                    packed = packed.permute(0, 2, 4, 1, 3, 5)
                    compressed_latent_pred = packed.reshape(batch_size, num_patches, channels)

                    reconstructed_velocity_pred = noise - compressed_latent_pred
                    loss = F.mse_loss(reconstructed_velocity_pred, velocity_target)
                    results[t_idx_int][compressed_size].append(loss.item())

    return results


def run_experiment_auraflow(pipe, dataset_samples, use_caption, config):
    """Run experiment with AuraFlow model."""
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=config.device)
    pipe.transformer.eval()

    results = {t.item(): {**{size: [] for size in config.compression_sizes}, 'baseline': []} for t in config.t_indices}

    for sample in tqdm(dataset_samples, desc=f"AuraFlow (caption={use_caption})"):
        with torch.no_grad():
            prompt = _get_caption(sample, use_caption)
            prompt_embeds = encode_prompt_for_model(pipe, "auraflow", prompt, config.device, config.dtype)

            image = _preprocess_image(sample, config.width, config.height, config.device, config.dtype)

            latents = pipe.vae.encode(image).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)

            for t_idx in config.t_indices:
                t_idx_int = t_idx.item()
                sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=config.dtype)

                noisy_latents = (1 - sigma) * latents + sigma * noise
                velocity_target = noise - latents

                timestep = pipe.scheduler.timesteps[t_idx_int].unsqueeze(0).to(config.device)

                velocity_pred = pipe.transformer(
                    hidden_states=noisy_latents,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False
                )[0]

                baseline_loss = F.mse_loss(velocity_pred, velocity_target)
                results[t_idx_int]['baseline'].append(baseline_loss.item())

                latent_pred = noise - velocity_pred

                for compressed_size in config.compression_sizes:
                    compressed_latent_pred = apply_spatial_compression(latent_pred, compressed_size)
                    reconstructed_velocity_pred = noise - compressed_latent_pred
                    loss = F.mse_loss(reconstructed_velocity_pred, velocity_target)
                    results[t_idx_int][compressed_size].append(loss.item())

    return results


_EXPERIMENT_RUNNERS = {
    "sd3": run_experiment_sd3,
    "flux": run_experiment_flux,
    "auraflow": run_experiment_auraflow,
}


def run_all_experiments(experiment_configs, dataset_samples, config):
    """Run all experiment configurations and return results dict."""
    all_results = {}

    for model_name, use_caption in experiment_configs:
        experiment_key = f"{model_name}_caption={use_caption}"
        print(f"\n{'='*50}")
        print(f"Running experiment: {experiment_key}")
        print(f"{'='*50}")

        pipe = load_model(model_name, config.device, config.dtype)

        runner = _EXPERIMENT_RUNNERS[model_name]
        results = runner(pipe, dataset_samples, use_caption, config)
        all_results[experiment_key] = results

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll experiments completed!")
    return all_results
