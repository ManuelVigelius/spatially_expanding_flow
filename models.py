import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, FluxPipeline, AuraFlowPipeline, DiTPipeline


def load_model(model_name, device, dtype):
    if model_name == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=dtype
        )
    elif model_name == "flux":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype
        )
    elif model_name == "auraflow":
        pipe = AuraFlowPipeline.from_pretrained(
            "fal/AuraFlow-v0.3",
            torch_dtype=dtype
        )
    elif model_name == "dit":
        pipe = DiTPipeline.from_pretrained(
            "facebook/DiT-XL-2-512",
            torch_dtype=dtype
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return pipe.to(device)


def encode_image(pipe, model_name, image):
    """VAE-encode an image to latents in [B, C, H, W] format (never packed).

    Args:
        image: Normalised pixel tensor [B, 3, H, W] in [-1, 1].

    Returns:
        latents: [B, C, H, W]
    """
    latents = pipe.vae.encode(image).latent_dist.sample()
    if model_name in ("sd3", "flux"):
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    elif model_name in ("auraflow", "dit"):
        latents = latents * pipe.vae.config.scaling_factor
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return latents


def encode_prompt(pipe, model_name, prompt, device, dtype):
    """Encode a conditioning input into an opaque prompt_data dict.

    For text-conditioned models (sd3, flux, auraflow), `prompt` is a string.
    For DiT, `prompt` is an integer ImageNet class label (or a list of them
    for a batch); pass the null class (1000) for unconditional generation.

    Returns a dict that can be passed directly to `predict`.
    """
    with torch.no_grad():
        if model_name == "sd3":
            embeds, _, pooled, _ = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            return {"embeds": embeds, "pooled": pooled}

        elif model_name == "flux":
            embeds, pooled, txt_ids = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt,
                device=device, num_images_per_prompt=1,
            )
            return {"embeds": embeds, "pooled": pooled, "txt_ids": txt_ids}

        elif model_name == "auraflow":
            text_inputs = pipe.tokenizer(
                prompt, padding="max_length", max_length=256,
                truncation=True, return_tensors="pt",
            )
            out = pipe.text_encoder(text_inputs.input_ids.to(device), output_hidden_states=False)
            embeds = (out[0] if isinstance(out, tuple) else out.last_hidden_state).to(dtype=dtype)
            return {"embeds": embeds}

        elif model_name == "dit":
            labels = prompt if isinstance(prompt, torch.Tensor) else torch.tensor(
                prompt if isinstance(prompt, list) else [prompt], device=device
            )
            return {"class_labels": labels.to(device)}

        else:
            raise ValueError(f"Unknown model: {model_name}")


def predict(pipe, model_name, latents, timestep, prompt_data, guidance_scale=1.0):
    """Run one transformer forward pass and return the velocity in [B, C, H, W].

    All packing/unpacking for FLUX is handled internally.

    Args:
        latents:        [B, C, H, W] — never packed.
        timestep:       scalar or 0-d tensor on the correct device/dtype.
        prompt_data:    dict returned by encode_prompt. For a single-prompt dict,
                        embeddings are broadcast to the batch size automatically.
        guidance_scale: CFG scale. >1.0 enables classifier-free guidance via a
                        doubled forward pass (cond + uncond). The uncond prompt
                        is the null class (DiT) or an empty-string encoding
                        (sd3, auraflow). Ignored for flux, which uses distilled
                        guidance internally.

    Returns:
        velocity: [B, C, H, W]
    """
    B = latents.shape[0]
    device = latents.device
    do_cfg = guidance_scale > 1.0

    if model_name == "sd3":
        embeds = prompt_data["embeds"].expand(B, -1, -1)
        pooled = prompt_data["pooled"].expand(B, -1)
        if do_cfg:
            uncond_embeds, _, uncond_pooled, _ = pipe.encode_prompt(
                prompt="", prompt_2="", prompt_3="",
                device=device, num_images_per_prompt=B,
                do_classifier_free_guidance=False,
            )
            embeds_in = torch.cat([uncond_embeds, embeds])
            pooled_in = torch.cat([uncond_pooled, pooled])
            latents_in = latents.repeat(2, 1, 1, 1)
        else:
            embeds_in, pooled_in, latents_in = embeds, pooled, latents
        t = timestep.unsqueeze(0).to(device) if timestep.dim() == 0 else timestep.to(device)
        out = pipe.transformer(
            hidden_states=latents_in, timestep=t,
            encoder_hidden_states=embeds_in, pooled_projections=pooled_in,
            return_dict=False,
        )[0]
        if do_cfg:
            vel_uncond, vel_cond = out.chunk(2)
            return vel_uncond + guidance_scale * (vel_cond - vel_uncond)
        return out

    elif model_name == "flux":
        _, _, lh, lw = latents.shape
        embeds = prompt_data["embeds"].expand(B, -1, -1)
        pooled = prompt_data["pooled"].expand(B, -1)
        txt_ids = prompt_data["txt_ids"]
        guidance = torch.tensor([3.5], device=device, dtype=latents.dtype)
        img_ids = pipe._prepare_latent_image_ids(1, lh // 2, lw // 2, device, latents.dtype)

        packed = pipe._pack_latents(latents, B, pipe.transformer.config.in_channels // 4, lh, lw)
        t = timestep.expand(B).to(device) / 1000.0
        vel_packed = pipe.transformer(
            hidden_states=packed, timestep=t, guidance=guidance,
            encoder_hidden_states=embeds, pooled_projections=pooled,
            txt_ids=txt_ids, img_ids=img_ids, return_dict=False,
        )[0]
        return _unpack_latents(vel_packed, lh, lw)

    elif model_name == "auraflow":
        embeds = prompt_data["embeds"].expand(B, -1, -1)
        if do_cfg:
            uncond_inputs = pipe.tokenizer(
                "", padding="max_length", max_length=256,
                truncation=True, return_tensors="pt",
            )
            uncond_out = pipe.text_encoder(uncond_inputs.input_ids.to(device), output_hidden_states=False)
            uncond_embeds = (uncond_out[0] if isinstance(uncond_out, tuple) else uncond_out.last_hidden_state)
            uncond_embeds = uncond_embeds.to(dtype=latents.dtype).expand(B, -1, -1)
            embeds_in = torch.cat([uncond_embeds, embeds])
            latents_in = latents.repeat(2, 1, 1, 1)
        else:
            embeds_in, latents_in = embeds, latents
        t = timestep.unsqueeze(0).to(device) / 1000 if timestep.dim() == 0 else timestep.to(device) / 1000
        out = pipe.transformer(
            hidden_states=latents_in, timestep=t,
            encoder_hidden_states=embeds_in, return_dict=False,
        )[0]
        if do_cfg:
            vel_uncond, vel_cond = out.chunk(2)
            return vel_uncond + guidance_scale * (vel_cond - vel_uncond)
        return out

    elif model_name == "dit":
        NULL_CLASS = 1000
        class_labels = prompt_data["class_labels"].expand(B)
        if do_cfg:
            uncond_labels = torch.full((B,), NULL_CLASS, device=device, dtype=torch.long)
            labels_in = torch.cat([uncond_labels, class_labels])
            latents_in = latents.repeat(2, 1, 1, 1)
        else:
            labels_in, latents_in = class_labels, latents
        out = pipe.transformer(
            latents_in, timestep=timestep.expand(latents_in.shape[0]).to(device),
            class_labels=labels_in, return_dict=False,
        )[0]
        noise_pred = out.chunk(2, dim=1)[0]  # discard predicted variance
        if do_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            return noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        return noise_pred

    else:
        raise ValueError(f"Unknown model: {model_name}")


def downsample_latents(latents, size, mode="area"):
    """Downsample latents [B, C, H, W] to the given square spatial size.

    No-op if the latents are already at that size. Uses area interpolation by
    default; pass mode="bilinear" for bilinear interpolation.
    """
    if latents.shape[2] == size and latents.shape[3] == size:
        return latents
    kwargs = {} if mode == "area" else {"align_corners": True}
    return F.interpolate(latents, size=(size, size), mode=mode, **kwargs)


def upsample_latents(latents, size):
    """Bilinearly upsample latents [B, C, H, W] to the given spatial size.

    Args:
        size: int (square) or (h, w) tuple.

    No-op if the latents are already at that size.
    """
    if isinstance(size, int):
        size = (size, size)
    if latents.shape[2:] == torch.Size(size):
        return latents
    return F.interpolate(latents, size=size, mode="bilinear", align_corners=True)


def apply_spatial_compression(latents, target_size, downsample_mode="area"):
    """Downsample then upsample latents to simulate a virtual resize."""
    original_size = latents.shape[2:]
    return upsample_latents(downsample_latents(latents, target_size, mode=downsample_mode), original_size)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _unpack_latents(packed, lh, lw):
    """Inverse of FluxPipeline._pack_latents: [B, N, C] -> [B, ch, lh, lw]."""
    B, N, C = packed.shape
    ch = C // 4
    return packed.view(B, lh // 2, lw // 2, ch, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, ch, lh, lw)
