import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models import load_model, encode_prompt_for_model, apply_spatial_compression


def _get_caption(sample, use_caption):
    if use_caption and "captions" in sample and sample["captions"]:
        captions = sample["captions"]
        return captions[0] if isinstance(captions, list) else captions
    return ""


def _preprocess_image(sample, width, height, device, dtype):
    image = sample["image"].convert("RGB")
    image = image.resize((width, height))
    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device=device, dtype=dtype)
    return (image / 127.5) - 1.0


def _eval_compression(noise, velocity_pred, velocity_target, compression_sizes, compress_fn, vel_results, lat_results, t_idx_int):
    """Shared baseline + compression evaluation for a single timestep."""
    vel_results[t_idx_int]['baseline'].append(F.mse_loss(velocity_pred, velocity_target).item())
    latent_pred = noise - velocity_pred
    for size in compression_sizes:
        compressed = compress_fn(latent_pred, size)
        vel_results[t_idx_int][size].append(F.mse_loss(noise - compressed, velocity_target).item())
        lat_results[t_idx_int][size].append(F.mse_loss(compressed, latent_pred).item())


def _init_results(config, include_baseline=True):
    def make():
        d = {s: [] for s in config.compression_sizes}
        if include_baseline:
            d['baseline'] = []
        return d
    return {t.item(): make() for t in config.t_indices}


def _sample_loop(pipe, model_name, dataset_samples, use_caption, config, encode_fn, denoise_fn, compress_fn):
    """Generic sample loop: encode prompt/image, iterate timesteps, evaluate compression."""
    vel_results = _init_results(config)
    lat_results = _init_results(config, include_baseline=False)
    for sample in tqdm(dataset_samples, desc=f"{model_name} (caption={use_caption})"):
        with torch.no_grad():
            prompt = _get_caption(sample, use_caption)
            prompt_data = encode_fn(prompt)
            image = _preprocess_image(sample, config.width, config.height, config.device, config.dtype)
            latents = pipe.vae.encode(image).latent_dist.sample()
            working, noise = denoise_fn(latents)

            for t_idx in config.t_indices:
                t_idx_int = t_idx.item()
                sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=config.dtype)
                noisy = (1 - sigma) * working + sigma * noise
                target = noise - working
                timestep = pipe.scheduler.timesteps[t_idx_int]
                vel = prompt_data(noisy, timestep)
                _eval_compression(noise, vel, target, config.compression_sizes, compress_fn, vel_results, lat_results, t_idx_int)
    return {"velocity": vel_results, "latent": lat_results}


def run_experiment_sd3(pipe, dataset_samples, use_caption, config):
    pipe.scheduler.set_timesteps(config.num_inference_steps)
    pipe.transformer.eval()

    def encode_fn(prompt):
        embeds, pooled = encode_prompt_for_model(pipe, "sd3", prompt, config.device, config.dtype)
        def predict(noisy, t):
            return pipe.transformer(hidden_states=noisy, timestep=t.unsqueeze(0).to(config.device),
                                    encoder_hidden_states=embeds, pooled_projections=pooled, return_dict=False)[0]
        return predict

    def denoise_fn(latents):
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        return latents, torch.randn_like(latents)

    return _sample_loop(pipe, "SD3", dataset_samples, use_caption, config, encode_fn, denoise_fn,
                        lambda pred, s: apply_spatial_compression(pred, s))


def run_experiment_flux(pipe, dataset_samples, use_caption, config):
    pipe.transformer.eval()
    lh, lw = config.height // 8, config.width // 8
    ph, pw = lh // 2, lw // 2

    seq_len = ph * pw
    sc = pipe.scheduler.config
    mu = sc.base_shift + sc.max_shift * ((seq_len - sc.base_image_seq_len) / (sc.max_image_seq_len - sc.base_image_seq_len))
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=config.device, mu=mu)

    guidance = torch.tensor([3.5], device=config.device, dtype=config.dtype)
    img_ids = pipe._prepare_latent_image_ids(1, ph, pw, config.device, config.dtype)

    def encode_fn(prompt):
        embeds, pooled, txt_ids = encode_prompt_for_model(pipe, "flux", prompt, config.device, config.dtype)
        def predict(noisy, t):
            return pipe.transformer(hidden_states=noisy, timestep=t.expand(1).to(config.device) / 1000.0,
                                    guidance=guidance, encoder_hidden_states=embeds, pooled_projections=pooled,
                                    txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
        return predict

    def denoise_fn(latents):
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        packed = pipe._pack_latents(latents, 1, pipe.transformer.config.in_channels // 4, lh, lw)
        return packed, torch.randn_like(packed)

    def compress_fn(latent_pred, size):
        B, N, C = latent_pred.shape
        h, w, ch = config.height // pipe.vae_scale_factor, config.width // pipe.vae_scale_factor, C // 4
        spatial = latent_pred.view(B, h//2, w//2, ch, 2, 2).permute(0,3,1,4,2,5).reshape(B, ch, h, w)
        compressed = apply_spatial_compression(spatial, size)
        return compressed.view(B, ch, h//2, 2, w//2, 2).permute(0,2,4,1,3,5).reshape(B, N, C)

    return _sample_loop(pipe, "FLUX", dataset_samples, use_caption, config, encode_fn, denoise_fn, compress_fn)


def run_experiment_auraflow(pipe, dataset_samples, use_caption, config):
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=config.device)
    pipe.transformer.eval()

    def encode_fn(prompt):
        embeds = encode_prompt_for_model(pipe, "auraflow", prompt, config.device, config.dtype)
        def predict(noisy, t):
            return pipe.transformer(hidden_states=noisy, timestep=t.unsqueeze(0).to(config.device) / 1000,
                                    encoder_hidden_states=embeds, return_dict=False)[0]
        return predict

    def denoise_fn(latents):
        latents = latents * pipe.vae.config.scaling_factor
        return latents, torch.randn_like(latents)

    return _sample_loop(pipe, "AuraFlow", dataset_samples, use_caption, config, encode_fn, denoise_fn,
                        lambda pred, s: apply_spatial_compression(pred, s))


_RUNNERS = {"sd3": run_experiment_sd3, "flux": run_experiment_flux, "auraflow": run_experiment_auraflow}


def run_all_experiments(experiment_configs, dataset_samples, config):
    all_results = {}
    for model_name, use_caption in experiment_configs:
        key = f"{model_name}_caption={use_caption}"
        print(f"\n{'='*50}\nRunning: {key}\n{'='*50}")
        pipe = load_model(model_name, config.device, config.dtype)
        all_results[key] = _RUNNERS[model_name](pipe, dataset_samples, use_caption, config)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nAll experiments completed!")
    return all_results
