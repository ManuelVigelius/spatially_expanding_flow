"""
Reproducible noise generator for SD3 latents at 512x512.

Generates noise for each image sequentially (one at a time) so that the
noise for image i is always identical regardless of batch size.
"""

import torch


_LATENT_H = 64  # 512 // 8
_LATENT_W = 64


def get_img_noise(batch_size: int, start_index: int = 0, device="cpu", latent_c: int = 4) -> torch.Tensor:
    """
    Generate reproducible noise for a batch of images.

    Each image's noise is generated independently with seed = start_index + i,
    so the noise for image i is always the same regardless of batch_size.

    Args:
        batch_size: Number of noise tensors to return.
        start_index: Global index of the first image in this batch (default 0).
        device: Target device for the returned tensor.
        latent_c: Number of latent channels (4 for SDXL, 16 for SD3).

    Returns:
        Float32 noise tensor of shape [batch_size, latent_c, 64, 64].
        Cast to bfloat16 before saving if needed.
    """
    noise_list = []
    for i in range(batch_size):
        seed = start_index + i
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = torch.randn(
            1, latent_c, _LATENT_H, _LATENT_W,
            generator=generator,
            dtype=torch.float32,
        )
        noise_list.append(noise)
    return torch.cat(noise_list, dim=0).to(device)
