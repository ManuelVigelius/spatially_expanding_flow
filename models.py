import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, FluxPipeline, AuraFlowPipeline


def apply_spatial_compression(predicted_x0, target_size):
    """
    Downsample and upsample the predicted clean image.

    Args:
        predicted_x0: Predicted clean latent [B, C, H, W]
        target_size: Target spatial size (int)

    Returns:
        Compressed and upsampled predicted_x0
    """
    target_size = (target_size, target_size)
    original_size = predicted_x0.shape[2:]

    compressed = F.interpolate(
        predicted_x0,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )
    assert compressed.shape[2:] == target_size

    upsampled = F.interpolate(
        compressed,
        size=original_size,
        mode='bilinear',
        align_corners=False
    )

    return upsampled


def load_model(model_name, device, dtype):
    """Load a diffusion model by name."""
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return pipe.to(device)


def encode_prompt_for_model(pipe, model_name, prompt, device, dtype):
    """Encode a prompt for the given model."""
    with torch.no_grad():
        if model_name == "sd3":
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            return prompt_embeds, pooled_prompt_embeds
        elif model_name == "flux":
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
            )
            return prompt_embeds, pooled_prompt_embeds, text_ids
        elif model_name == "auraflow":
            text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)

            text_encoder_output = pipe.text_encoder(text_input_ids, output_hidden_states=False)
            prompt_embeds = text_encoder_output[0] if isinstance(text_encoder_output, tuple) else text_encoder_output.last_hidden_state
            prompt_embeds = prompt_embeds.to(dtype=dtype)
            return prompt_embeds
    raise ValueError(f"Unknown model: {model_name}")
