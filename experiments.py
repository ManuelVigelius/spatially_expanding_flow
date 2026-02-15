from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models import load_model, encode_prompt_for_model, apply_spatial_compression


def _process_sample(sample, use_caption, width, height, device, dtype):
    if use_caption and "captions" in sample and sample["captions"]:
        captions = sample["captions"]
        caption = captions[0] if isinstance(captions, list) else captions
    else:
        caption = ""
    image = sample["image"].convert("RGB")
    image = image.resize((width, height))
    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device=device, dtype=dtype)
    image = (image / 127.5) - 1.0
    return caption, image


def _sample_loop(pipe, model_name, dataset_samples, use_caption, config, encode_fn, prepare_latents_fn, compress_fn):
    """Generic sample loop: encode prompt/image, iterate timesteps, evaluate compression."""
    vel_results = defaultdict(lambda: defaultdict(list))
    lat_results = defaultdict(lambda: defaultdict(list))
    for sample in tqdm(dataset_samples, desc=f"{model_name} (caption={use_caption})"):
        with torch.no_grad():
            caption, image = _process_sample(sample, use_caption, config.width, config.height, config.device, config.dtype)
            prompt_data = encode_fn(caption)
            raw_latents = pipe.vae.encode(image).latent_dist.sample()
            latents = prepare_latents_fn(raw_latents)

            for t_idx in config.t_indices:
                t_idx_int = t_idx.item()
                sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=config.dtype)
                timestep = pipe.scheduler.timesteps[t_idx_int]

                for size in config.compression_sizes:
                    noise = torch.randn_like(latents)
                    noisy = (1 - sigma) * compress_fn(latents, size) + sigma * noise
                    target = noise - latents
                    vel = prompt_data(noisy, timestep)
                    latent_pred = noisy - vel * sigma
                    noise_pred = noisy + vel * (1 - sigma)

                    compressed = compress_fn(latent_pred, size)
                    # using the clean noise yields results very similar to only predicting the clean latents
                    # as the compression removes most of the noise
                    vel_results[t_idx_int][size].append(F.mse_loss(noise_pred - compressed, target).item())
                    lat_results[t_idx_int][size].append(F.mse_loss(compressed, latents).item())
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

    def prepare_latents_fn(latents):
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        return latents

    return _sample_loop(pipe, "SD3", dataset_samples, use_caption, config, encode_fn, prepare_latents_fn,
                        apply_spatial_compression)


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

    def prepare_latents_fn(latents):
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        packed = pipe._pack_latents(latents, 1, pipe.transformer.config.in_channels // 4, lh, lw)
        return packed

    packed_shape = (lh, lw)
    return _sample_loop(pipe, "FLUX", dataset_samples, use_caption, config, encode_fn, prepare_latents_fn,
                        partial(apply_spatial_compression, packed_shape=packed_shape))


def run_experiment_auraflow(pipe, dataset_samples, use_caption, config):
    pipe.scheduler.set_timesteps(config.num_inference_steps, device=config.device)
    pipe.transformer.eval()

    def encode_fn(prompt):
        embeds = encode_prompt_for_model(pipe, "auraflow", prompt, config.device, config.dtype)
        def predict(noisy, t):
            return pipe.transformer(hidden_states=noisy, timestep=t.unsqueeze(0).to(config.device) / 1000,
                                    encoder_hidden_states=embeds, return_dict=False)[0]
        return predict

    def prepare_latents_fn(latents):
        latents = latents * pipe.vae.config.scaling_factor
        return latents

    return _sample_loop(pipe, "AuraFlow", dataset_samples, use_caption, config, encode_fn, prepare_latents_fn,
                        apply_spatial_compression)


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
