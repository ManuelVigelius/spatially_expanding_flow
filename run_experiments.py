"""
Unified experiment runner.

Runs any combination of three experiment modes:
  --resize           Test model performance across image sizes (real downsampling).
  --virtual-resize   Test virtual spatial compression across models.
  --virtual-vs-real  Compare virtual resize vs. real resize predictions.

If no mode flag is given, all three are run.

Examples:
    python run_experiments.py --resize --num-samples 4 --image-sizes 64 128
    python run_experiments.py --virtual-resize --vr-models sd3
    python run_experiments.py --virtual-vs-real --vvr-sizes 64 128 256
    python run_experiments.py --output results.pkl   # runs all three
"""

import argparse

import torch
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

from models import (
    load_model, encode_image, encode_prompt, predict,
    downsample_latents, upsample_latents, apply_spatial_compression,
)
from experiment_utils import (
    load_samples, image_to_tensor, encode_batch,
    setup_scheduler, is_flow_matching, get_sigmas, get_alphas,
    make_noisy, scale_sigma, mse,
    nested_defaultdict, defaultdict_to_dict, save_results,
    denoise_step_flow, denoise_step_ddpm, decode_latents,
)


# --------------------------------------------------------------------------- #
# Experiment: resize
# --------------------------------------------------------------------------- #

def _get_prompt_dit(batch):
    """DiT uses integer class labels; other models use text strings."""
    return [s["label"] if s["label"] is not None else 1000 for s in batch]


def run_resize(pipe, dataset_samples, args):
    """
    For every image size and every timestep, downsample 512-latents to `size`,
    add noise, run the model, and record latent MSE and upsampled-latent MSE.

    Returns:
        {
            "latent":           {size: {t_idx: [float, ...]}},
            "upsampled_latent": {size: {t_idx: [float, ...]}},
            "meta":             {hyperparams},
        }
    """
    t_indices = torch.linspace(0, args.num_inference_steps - 1, args.num_t_indices).long()
    image_sizes = args.image_sizes

    setup_scheduler(pipe, args.model, args.num_inference_steps, device)
    pipe.transformer.eval()
    flow_matching = is_flow_matching(pipe.scheduler)
    if flow_matching:
        sigmas = get_sigmas(pipe.scheduler)
    else:
        sqrt_alphas, sqrt_one_minus_alphas = get_alphas(pipe.scheduler, device, dtype)

    lat_results = nested_defaultdict()
    upl_results = nested_defaultdict()

    batches = [dataset_samples[i: i + args.batch_size] for i in range(0, len(dataset_samples), args.batch_size)]

    for size in tqdm(image_sizes, desc=f"{args.model} resize experiment"):
        with torch.no_grad():
            for batch in batches:
                B = len(batch)
                prompt = _get_prompt_dit(batch) if args.model == "dit" else ""
                prompt_data = encode_prompt(pipe, args.model, prompt, device, dtype)

                ref_latents = encode_batch(pipe, args.model, batch, 512, device, dtype)
                latent_spatial = ref_latents.shape[2:]
                latents = downsample_latents(ref_latents, size // 8)

                for t_idx in t_indices:
                    t_idx_int = t_idx.item()
                    noise = torch.randn_like(latents)

                    if flow_matching:
                        scale_fn = (lambda s: scale_sigma(s, size, 512)) if args.scale_sigma else None
                        latent_pred, _ = denoise_step_flow(
                            pipe, args.model, latents, noise, t_idx_int, prompt_data,
                            guidance_scale=args.cfg_scale, scale_sigma_fn=scale_fn,
                        )
                    else:
                        latent_pred = denoise_step_ddpm(
                            pipe, args.model, latents, noise, t_idx_int,
                            sqrt_alphas, sqrt_one_minus_alphas, prompt_data,
                            guidance_scale=args.cfg_scale,
                        )

                    for b in range(B):
                        lat_results[size][t_idx_int].append(mse(latent_pred[b], latents[b]))

                    upsampled_pred = upsample_latents(latent_pred, latent_spatial)
                    for b in range(B):
                        upl_results[size][t_idx_int].append(mse(upsampled_pred[b], ref_latents[b]))

    return {
        "latent":           defaultdict_to_dict(lat_results),
        "upsampled_latent": defaultdict_to_dict(upl_results),
        "meta": {
            "model": args.model, "dataset": args.dataset,
            "scale_sigma": args.scale_sigma, "cfg_scale": args.cfg_scale,
            "image_sizes": image_sizes, "num_samples": args.num_samples,
        },
    }


# --------------------------------------------------------------------------- #
# Experiment: virtual resize
# --------------------------------------------------------------------------- #

def run_virtual_resize(pipe, model_name, dataset_samples, args):
    """
    For each sample, timestep, and compression size: apply virtual spatial
    compression to latents, add noise, run model, and record 6 metrics.

    The input latent is always compressed to `size` before noising. The three
    conditions differ in how prediction and targets are handled:

      compress_output:  re-compress predicted latent to `size`, compare vs full-res GT
      compress_targets: compare raw full-res prediction vs downsampled GT at `size`
      no_compress_output: compare raw full-res prediction vs full-res GT

    When size == full_size of the latent, all resizes are no-ops and these
    conditions collapse to the uncompressed baseline.

    Flow-matching models (sd3, flux, auraflow):
        velocity target = noise - latents
        latent_pred = noisy - vel * sigma

    DDPM models (dit):
        velocity target = noise  (epsilon)
        latent_pred = (noisy - sqrt_one_minus_alpha * eps) / sqrt_alpha

    Returns:
        {
            "velocity_compress_output":   {t_idx: {size: [float, ...]}},
            "latent_compress_output":     {t_idx: {size: [float, ...]}},
            "velocity_compress_targets":  {t_idx: {size: [float, ...]}},
            "latent_compress_targets":    {t_idx: {size: [float, ...]}},
            "velocity_no_compress":       {t_idx: {size: [float, ...]}},
            "latent_no_compress":         {t_idx: {size: [float, ...]}},
        }
    """
    t_indices = torch.linspace(0, args.num_inference_steps - 1, args.num_t_indices).long()

    setup_scheduler(pipe, model_name, args.num_inference_steps, device)
    pipe.transformer.eval()

    flow_matching = is_flow_matching(pipe.scheduler)
    if not flow_matching:
        sqrt_alphas, sqrt_one_minus_alphas = get_alphas(pipe.scheduler, device, dtype)

    vel_comp_out  = nested_defaultdict()
    lat_comp_out  = nested_defaultdict()
    vel_comp_tgt  = nested_defaultdict()
    lat_comp_tgt  = nested_defaultdict()
    vel_no_comp   = nested_defaultdict()
    lat_no_comp   = nested_defaultdict()
    # zero-prediction baselines (same conditions, predicted vel/noise = 0)
    vel_zero_comp_out  = nested_defaultdict()
    lat_zero_comp_out  = nested_defaultdict()
    vel_zero_comp_tgt  = nested_defaultdict()
    lat_zero_comp_tgt  = nested_defaultdict()
    vel_zero_no_comp   = nested_defaultdict()
    lat_zero_no_comp   = nested_defaultdict()

    for sample in tqdm(dataset_samples, desc=f"{model_name} virtual resize"):
        with torch.no_grad():
            if model_name == "dit":
                prompt = sample["label"] if sample["label"] is not None else 1000
            elif args.use_captions and "captions" in sample and sample["captions"]:
                captions = sample["captions"]
                prompt = captions[0] if isinstance(captions, list) else captions
            else:
                prompt = ""

            image = image_to_tensor(sample["image"], 512, device, dtype)
            prompt_data = encode_prompt(pipe, model_name, prompt, device, dtype)
            latents = encode_image(pipe, model_name, image)

            for t_idx in t_indices:
                t_idx_int = t_idx.item()
                noise = torch.randn_like(latents)

                if flow_matching:
                    sigma = pipe.scheduler.sigmas[t_idx_int].to(dtype=dtype)
                    timestep = pipe.scheduler.timesteps[t_idx_int]
                    vel_target = noise - latents
                else:
                    sa = sqrt_alphas[t_idx_int]
                    sb = sqrt_one_minus_alphas[t_idx_int]
                    timestep = pipe.scheduler.timesteps[t_idx_int]
                    vel_target = noise  # epsilon target

                for size in args.compression_sizes:
                    compressed = apply_spatial_compression(latents, size)

                    if flow_matching:
                        noisy = make_noisy(compressed, noise, sigma)
                        vel = predict(pipe, model_name, noisy, timestep, prompt_data)
                        latent_pred = noisy - vel * sigma
                        noise_pred  = noisy + vel * (1 - sigma)
                        # zero-prediction baselines: vel=0 -> latent_pred=noisy, noise_pred=noisy
                        latent_zero = noisy
                        noise_zero  = noisy
                    else:
                        noisy = sa * compressed + sb * noise
                        eps = predict(pipe, model_name, noisy, timestep, prompt_data)
                        latent_pred = (noisy - sb * eps) / sa
                        noise_pred  = eps
                        # zero-prediction baselines: eps=0 -> latent_pred=noisy/sa, noise_pred=0
                        latent_zero = noisy / sa
                        noise_zero  = torch.zeros_like(eps)

                    vel_target_down  = downsample_latents(vel_target, size)
                    latents_down     = downsample_latents(latents, size)

                    # condition 1: re-compress output, compare vs full-res GT
                    latent_pred_comp = apply_spatial_compression(latent_pred, size)
                    latent_zero_comp = apply_spatial_compression(latent_zero, size)
                    vel_comp_out[t_idx_int][size].append(mse(noise_pred, vel_target))
                    lat_comp_out[t_idx_int][size].append(mse(latent_pred_comp, latents))
                    vel_zero_comp_out[t_idx_int][size].append(mse(noise_zero, vel_target))
                    lat_zero_comp_out[t_idx_int][size].append(mse(latent_zero_comp, latents))

                    # condition 2: compress targets, compare downsampled output vs downsampled GT
                    noise_pred_down  = downsample_latents(noise_pred, size)
                    latent_pred_down = downsample_latents(latent_pred, size)
                    noise_zero_down  = downsample_latents(noise_zero, size)
                    latent_zero_down = downsample_latents(latent_zero, size)
                    vel_comp_tgt[t_idx_int][size].append(mse(noise_pred_down, vel_target_down))
                    lat_comp_tgt[t_idx_int][size].append(mse(latent_pred_down, latents_down))
                    vel_zero_comp_tgt[t_idx_int][size].append(mse(noise_zero_down, vel_target_down))
                    lat_zero_comp_tgt[t_idx_int][size].append(mse(latent_zero_down, latents_down))

                    # condition 3: no output compression, compare raw output vs full-res GT
                    vel_no_comp[t_idx_int][size].append(mse(noise_pred, vel_target))
                    lat_no_comp[t_idx_int][size].append(mse(latent_pred, latents))
                    vel_zero_no_comp[t_idx_int][size].append(mse(noise_zero, vel_target))
                    lat_zero_no_comp[t_idx_int][size].append(mse(latent_zero, latents))

    return {
        "velocity_compress_output":       defaultdict_to_dict(vel_comp_out),
        "latent_compress_output":         defaultdict_to_dict(lat_comp_out),
        "velocity_compress_targets":      defaultdict_to_dict(vel_comp_tgt),
        "latent_compress_targets":        defaultdict_to_dict(lat_comp_tgt),
        "velocity_no_compress":           defaultdict_to_dict(vel_no_comp),
        "latent_no_compress":             defaultdict_to_dict(lat_no_comp),
        "velocity_zero_compress_output":  defaultdict_to_dict(vel_zero_comp_out),
        "latent_zero_compress_output":    defaultdict_to_dict(lat_zero_comp_out),
        "velocity_zero_compress_targets": defaultdict_to_dict(vel_zero_comp_tgt),
        "latent_zero_compress_targets":   defaultdict_to_dict(lat_zero_comp_tgt),
        "velocity_zero_no_compress":      defaultdict_to_dict(vel_zero_no_comp),
        "latent_zero_no_compress":        defaultdict_to_dict(lat_zero_no_comp),
    }


# --------------------------------------------------------------------------- #
# Experiment: virtual vs real resize
# --------------------------------------------------------------------------- #

def run_virtual_vs_real(pipe, dataset_samples, args):
    """
    For each image and timestep, compare:
      - real resize:    model runs at small resolution, result is upsampled
      - virtual resize: latents are blurred at full res, model runs at full res

    Returns:
        {
            size: {t_idx: [{"real_vs_gt": float, "virtual_vs_gt": float, "real_vs_virtual": float}, ...]},
            "meta": {hyperparams},
        }
    """
    t_indices = torch.linspace(0, args.num_inference_steps - 1, args.num_t_indices).long()
    full_size = args.full_size
    full_latent_size = full_size // 8

    setup_scheduler(pipe, "sd3", args.num_inference_steps, device)
    pipe.transformer.eval()

    with torch.no_grad():
        prompt_data = encode_prompt(pipe, "sd3", "", device, dtype)

    results = {size: {} for size in args.vvr_sizes}

    batches = [dataset_samples[i: i + args.vvr_batch_size] for i in range(0, len(dataset_samples), args.vvr_batch_size)]

    for size in tqdm(args.vvr_sizes, desc="virtual vs real sizes"):
        size_results = {}
        latent_small_h = size // 8

        for batch in tqdm(batches, desc=f"  batches (size={size})", leave=False):
            B = len(batch)
            with torch.no_grad():
                latents_full = encode_batch(pipe, "sd3", batch, full_size, device, dtype)
                assert latents_full.shape == (B, 16, full_latent_size, full_latent_size)

                latent_spatial = latents_full.shape[2:]
                latents_small = downsample_latents(latents_full, latent_small_h)
                latents_virtual = upsample_latents(downsample_latents(latents_full, latent_small_h), latent_spatial)

                for t_idx in t_indices:
                    t_idx_int = t_idx.item()
                    noise_small = torch.randn_like(latents_small)
                    noise_full = torch.randn_like(latents_full)

                    # real resize prediction
                    latent_pred_real, _ = denoise_step_flow(
                        pipe, "sd3", latents_small, noise_small, t_idx_int, prompt_data,
                        scale_sigma_fn=lambda s: scale_sigma(s, size, full_size),
                    )
                    latent_pred_real_up = upsample_latents(latent_pred_real, latent_spatial)
                    assert latent_pred_real_up.shape == latents_full.shape

                    # virtual resize prediction (full resolution, unscaled sigma)
                    latent_pred_virtual, _ = denoise_step_flow(
                        pipe, "sd3", latents_virtual, noise_full, t_idx_int, prompt_data,
                    )
                    latent_pred_virtual = upsample_latents(
                        downsample_latents(latent_pred_virtual, latent_small_h), latent_spatial
                    )
                    assert latent_pred_virtual.shape == latents_full.shape

                    entries = size_results.setdefault(t_idx_int, [])
                    for b in range(B):
                        entries.append({
                            "real_vs_gt":      mse(latent_pred_real_up[b], latents_full[b]),
                            "virtual_vs_gt":   mse(latent_pred_virtual[b], latents_full[b]),
                            "real_vs_virtual": mse(latent_pred_real_up[b], latent_pred_virtual[b]),
                        })

        results[size] = size_results

    results["meta"] = {
        "full_size": full_size, "sizes": args.vvr_sizes,
        "scale_sigma": True, "num_samples": args.num_samples,
    }
    return results


# --------------------------------------------------------------------------- #
# Experiment: generate with size schedules
# --------------------------------------------------------------------------- #

BUILTIN_SCHEDULES: dict[str, list[int]] = {
    "full":           [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
    "early_small":    [ 512,  512,  512,  512,  512, 1024, 1024, 1024, 1024, 1024],
    "gradual_mild":   [ 512,  512,  640,  640,  768,  768,  896,  896, 1024, 1024],
    "gradual_medium": [ 384,  384,  512,  512,  640,  768,  896,  896, 1024, 1024],
    "gradual_steep":  [ 256,  256,  256,  384,  512,  640,  768,  896, 1024, 1024],
    "burst_early":    [  64,  128,  256,  512, 1024, 1024, 1024, 1024, 1024, 1024],
    "burst_mid":      [  64,  128,  192,  384,  768, 1024, 1024, 1024, 1024, 1024],
}


def _save_grid(images, path: str, cols: int = 4) -> None:
    from PIL import Image
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (w * cols, h * rows))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    grid.save(path)


@torch.no_grad()
def run_generate_schedules(pipe, args):
    """
    Generate images with SD3 using named per-step size schedules.

    Each schedule is a list of pixel sizes (one per denoising step) controlling
    the virtual-resize blur applied to x0 at that step. "full" is a no-op
    (equivalent to the standard manual Euler loop).

    Saves individual PNGs and a grid image under args.gen_out_dir.
    """
    import os
    from tqdm import tqdm

    os.makedirs(args.gen_out_dir, exist_ok=True)

    setup_scheduler(pipe, "sd3", args.gen_steps, device)
    sigmas = pipe.scheduler.sigmas
    timesteps = pipe.scheduler.timesteps

    assert len(timesteps) == args.gen_steps, (
        f"Schedule length must equal --gen-steps ({args.gen_steps}); "
        f"got {len(timesteps)} timesteps."
    )

    prompt_data = encode_prompt(pipe, "sd3", args.gen_prompt, device, dtype)
    lat_size = args.gen_image_size // 8

    schedules = {name: BUILTIN_SCHEDULES[name] for name in args.gen_schedules
                 if name in BUILTIN_SCHEDULES}

    for sched_name, size_schedule in tqdm(schedules.items(), desc="schedules"):
        assert len(size_schedule) == args.gen_steps, (
            f"Schedule '{sched_name}' has {len(size_schedule)} entries but "
            f"--gen-steps is {args.gen_steps}."
        )

        torch.manual_seed(args.gen_seed)
        latents = sigmas[0] * torch.randn(
            args.gen_n_images, 16, lat_size, lat_size, device=device, dtype=dtype
        )
        noise = latents / sigmas[0]  # the initial pure noise, kept for vel reconstruction

        for step_idx, (t, sigma) in enumerate(zip(timesteps, sigmas[:-1])):
            sigma_next = sigmas[step_idx + 1]
            target_lat = size_schedule[step_idx] // 8

            vel = predict(pipe, "sd3", latents, t, prompt_data, guidance_scale=args.gen_cfg_scale)

            x0_pred = latents - sigma * vel
            x0_blurred = apply_spatial_compression(x0_pred, target_lat)
            vel_blurred = noise - x0_blurred
            latents = latents + (sigma_next - sigma) * vel_blurred

        images = decode_latents(pipe, latents)

        sched_dir = os.path.join(args.gen_out_dir, sched_name)
        os.makedirs(sched_dir, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(sched_dir, f"{i:04d}.png"))
        _save_grid(images, os.path.join(args.gen_out_dir, f"{sched_name}_grid.png"))
        print(f"  [{sched_name}] schedule={size_schedule}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Run resize / virtual-resize / virtual-vs-real experiments.")

    # Experiment selection
    p.add_argument("--resize",             action="store_true")
    p.add_argument("--virtual-resize",     action="store_true", dest="virtual_resize")
    p.add_argument("--virtual-vs-real",    action="store_true", dest="virtual_vs_real")
    p.add_argument("--generate-schedules", action="store_true", dest="generate_schedules")

    # Shared
    p.add_argument("--num-samples",          type=int, default=64)
    p.add_argument("--num-inference-steps",  type=int, default=50, dest="num_inference_steps")
    p.add_argument("--num-t-indices",        type=int, default=20, dest="num_t_indices")

    # Resize-specific
    p.add_argument("--model",       default="dit", choices=["dit", "sd3", "flux", "auraflow"])
    p.add_argument("--dataset",     default="imagenet1k", choices=["imagenet1k", "coco", "div2k"])
    p.add_argument("--batch-size",  type=int, default=32, dest="batch_size")
    p.add_argument("--no-scale-sigma", action="store_false", dest="scale_sigma")
    p.add_argument("--cfg-scale",   type=float, default=4.0, dest="cfg_scale")
    p.add_argument("--image-sizes", type=int, nargs="+", dest="image_sizes",
                   default=list(range(16, 513, 16)))

    # Virtual-resize-specific
    p.add_argument("--vr-models", nargs="+", dest="vr_models",
                   default=["sd3", "flux", "auraflow"],
                   choices=["sd3", "flux", "auraflow", "dit"])
    p.add_argument("--use-captions", action="store_true", dest="use_captions")
    p.add_argument("--vr-dataset-text", default="coco",
                   choices=["coco", "imagenet1k", "div2k"], dest="vr_dataset_text",
                   help="Dataset for text-conditioned VR models (sd3, flux, auraflow).")
    p.add_argument("--compression-sizes", type=int, nargs="+", dest="compression_sizes",
                   default=[64, 56, 48, 32, 16, 8, 4, 2, 1])

    # Virtual-vs-real-specific
    p.add_argument("--full-size",      type=int, default=1024, dest="full_size")
    p.add_argument("--vvr-batch-size", type=int, default=16,   dest="vvr_batch_size")
    p.add_argument("--vvr-sizes",      type=int, nargs="+",    dest="vvr_sizes",
                   default=[64, 128, 256, 512, 768, 1024])
    p.add_argument("--vvr-dataset",    default="div2k",
                   choices=["coco", "imagenet1k", "div2k"], dest="vvr_dataset")

    # Generate-schedules-specific
    p.add_argument("--gen-prompt",      default="A photo of a cat sitting on a windowsill", dest="gen_prompt")
    p.add_argument("--gen-n-images",    type=int, default=4,    dest="gen_n_images")
    p.add_argument("--gen-steps",       type=int, default=10,   dest="gen_steps")
    p.add_argument("--gen-cfg-scale",   type=float, default=7.0, dest="gen_cfg_scale")
    p.add_argument("--gen-image-size",  type=int, default=1024, dest="gen_image_size")
    p.add_argument("--gen-seed",        type=int, default=42,   dest="gen_seed")
    p.add_argument("--gen-out-dir",     default="results/sd3_baseline", dest="gen_out_dir")
    p.add_argument("--gen-schedules",   nargs="+", dest="gen_schedules",
                   default=list(BUILTIN_SCHEDULES.keys()),
                   choices=list(BUILTIN_SCHEDULES.keys()))

    # Output
    p.add_argument("--output",         default="results.pkl")
    p.add_argument("--separate-files", action="store_true", dest="separate_files")

    args = p.parse_args()
    # Default: run all if none specified
    if not (args.resize or args.virtual_resize or args.virtual_vs_real or args.generate_schedules):
        args.resize = args.virtual_resize = args.virtual_vs_real = args.generate_schedules = True
    return args


def main():
    args = parse_args()
    all_results = {}

    if args.resize:
        print(f"\n{'='*50}\nRunning: resize ({args.model} / {args.dataset})\n{'='*50}")
        dataset_samples = load_samples(args.dataset, args.num_samples)
        pipe = load_model(args.model, device, dtype)
        result = run_resize(pipe, dataset_samples, args)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        all_results["resize"] = result
        if args.separate_files:
            base = args.output.removesuffix(".pkl")
            save_results(result, f"{base}_resize.pkl")

    if args.virtual_resize:
        print(f"\n{'='*50}\nRunning: virtual resize\n{'='*50}")
        vr_samples_text = None
        vr_samples_imagenet = None
        vr_results = {}
        for model_name in args.vr_models:
            if model_name == "dit":
                if vr_samples_imagenet is None:
                    vr_samples_imagenet = load_samples("imagenet1k", args.num_samples)
                samples = vr_samples_imagenet
                key = model_name
            else:
                if vr_samples_text is None:
                    vr_samples_text = load_samples(args.vr_dataset_text, args.num_samples)
                samples = vr_samples_text
                key = f"{model_name}_caption={args.use_captions}"
            print(f"  Model: {model_name}")
            pipe = load_model(model_name, device, dtype)
            vr_results[key] = run_virtual_resize(pipe, model_name, samples, args)
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results["virtual_resize"] = vr_results
        if args.separate_files:
            base = args.output.removesuffix(".pkl")
            save_results(vr_results, f"{base}_virtual_resize.pkl")

    if args.virtual_vs_real:
        print(f"\n{'='*50}\nRunning: virtual vs real resize\n{'='*50}")
        vvr_dataset_samples = load_samples(args.vvr_dataset, args.num_samples)
        pipe = load_model("sd3", device, dtype)
        result = run_virtual_vs_real(pipe, vvr_dataset_samples, args)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        all_results["virtual_vs_real"] = result
        if args.separate_files:
            base = args.output.removesuffix(".pkl")
            save_results(result, f"{base}_virtual_vs_real.pkl")

    if args.generate_schedules:
        print(f"\n{'='*50}\nRunning: generate schedules\n{'='*50}")
        pipe = load_model("sd3", device, dtype)
        run_generate_schedules(pipe, args)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.separate_files and all_results:
        save_results(all_results, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
