from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pickle
from datasets import load_dataset

from models import load_model, encode_image, encode_prompt, predict, apply_spatial_compression
import config


def _process_sample(sample, use_caption, width, height, device, dtype):
    if use_caption and "captions" in sample and sample["captions"]:
        captions = sample["captions"]
        caption = captions[0] if isinstance(captions, list) else captions
    else:
        caption = ""
    image = sample["image"].convert("RGB").resize((width, height))
    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device=device, dtype=dtype)
    image = (image / 127.5) - 1.0
    return caption, image


def _sample_loop(pipe, model_name, dataset_samples, use_caption, cfg):
    vel_results = defaultdict(lambda: defaultdict(list))
    lat_results = defaultdict(lambda: defaultdict(list))

    for sample in tqdm(dataset_samples, desc=f"{model_name} (caption={use_caption})"):
        with torch.no_grad():
            caption, image = _process_sample(sample, use_caption, cfg.width, cfg.height, cfg.device, cfg.dtype)
            prompt_data = encode_prompt(pipe, model_name, caption, cfg.device, cfg.dtype)
            latents = encode_image(pipe, model_name, image)

            for t_idx in cfg.t_indices:
                t_idx_int = t_idx.item()
                sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=cfg.dtype)
                timestep = pipe.scheduler.timesteps[t_idx_int]

                for size in cfg.compression_sizes:
                    noise = torch.randn_like(latents)
                    compressed = apply_spatial_compression(latents, size)
                    noisy = (1 - sigma) * compressed + sigma * noise
                    target = noise - latents

                    vel = predict(pipe, model_name, noisy, timestep, prompt_data)
                    latent_pred = noisy - vel * sigma
                    noise_pred = noisy + vel * (1 - sigma)

                    compressed_pred = apply_spatial_compression(latent_pred, size)
                    vel_results[t_idx_int][size].append(F.mse_loss(noise_pred - compressed_pred, target).item())
                    lat_results[t_idx_int][size].append(F.mse_loss(compressed_pred, latents).item())

    return {"velocity": vel_results, "latent": lat_results}


def run_experiment(pipe, model_name, dataset_samples, use_caption, cfg):
    pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=cfg.device)
    if model_name == "flux":
        lh, lw = cfg.height // 8, cfg.width // 8
        ph, pw = lh // 2, lw // 2
        seq_len = ph * pw
        sc = pipe.scheduler.config
        mu = sc.base_shift + sc.max_shift * (
            (seq_len - sc.base_image_seq_len) / (sc.max_image_seq_len - sc.base_image_seq_len)
        )
        pipe.scheduler.set_timesteps(cfg.num_inference_steps, device=cfg.device, mu=mu)
    pipe.transformer.eval()
    return _sample_loop(pipe, model_name, dataset_samples, use_caption, cfg)


def run_all_experiments(experiment_configs, dataset_samples, cfg):
    all_results = {}
    for model_name, use_caption in experiment_configs:
        key = f"{model_name}_caption={use_caption}"
        print(f"\n{'='*50}\nRunning: {key}\n{'='*50}")
        pipe = load_model(model_name, cfg.device, cfg.dtype)
        all_results[key] = run_experiment(pipe, model_name, dataset_samples, use_caption, cfg)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nAll experiments completed!")
    return all_results


def main():
    dataset = load_dataset("detection-datasets/coco", split="val")
    dataset_samples = list(dataset.select(range(config.num_samples)))
    all_results = run_all_experiments(config.experiment_configs, dataset_samples, config)

    saveable = {k: {m: dict(v) for m, v in metrics.items()} for k, metrics in all_results.items()}
    with open(config.results_path, "wb") as f:
        pickle.dump(saveable, f)


if __name__ == '__main__':
    main()
