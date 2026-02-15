# Spatially Expanding Flow

## Motivation

Diffusion Transformers (DiTs) spend significant compute generating images at full spatial resolution throughout the entire denoising process. However, during the early diffusion steps, predicted images are inherently blurry and lack fine spatial detail. This suggests that running the model at full resolution during these steps is wasteful — predicting at a lower spatial resolution and bilinearly upsampling the result should closely approximate the full-resolution prediction.

**The core idea:** Start the denoising process at a low spatial resolution and progressively expand to the full resolution as finer details emerge in later steps. This could substantially reduce the computational cost of DiT-based generation without meaningful quality loss.

## What This Repo Does

This repository does **not** implement the spatially expanding approach itself. Instead, it provides **empirical evidence** supporting the idea by measuring how much prediction quality degrades when spatial information is artificially restricted.

Concretely, for each diffusion timestep and compression level, the experiment:

1. **Restricts the input** — the clean latent image is downsampled to a small spatial size and bilinearly upsampled back before being mixed with noise. This simulates a DiT that only has access to low-resolution spatial information.
2. **Restricts the output** — the model's predicted clean latent is downsampled and upsampled in the same way. This simulates a DiT that can only produce low-resolution predictions.
3. **Measures the error** — two MSE metrics are computed:
   - **Velocity MSE**: How much the velocity prediction degrades after spatial compression.
   - **Latent MSE**: How well the spatially compressed prediction matches the original clean latents.

The experiments are run across three flow matching models — **Stable Diffusion 3**, **FLUX.1**, and **AuraFlow** — on 64 images from the COCO validation set, with 9 compression sizes ranging from 64x64 down to 1x1 (in latent space).

## Results

### Latent MSE

![Latent MSE](Latent%20MSE.png)

### Velocity MSE

![Velocity MSE](Velocity%20MSE.png)

The plots show mean MSE with 95% confidence intervals across 20 evenly spaced timesteps (x-axis) for each compression size (colored lines). The **64x64 curve is the baseline** — since the latent resolution is 64x64, this represents the uncompressed condition with no spatial information lost.

**Early timesteps (left)** tolerate aggressive spatial compression with relatively small increases in error, consistent with the hypothesis that early predictions are spatially simple. This is the key result: for the early diffusion steps that the spatially expanding idea targets, all models behave exactly as expected.

**Later timesteps (right)** show rapidly increasing error under compression, confirming that fine spatial detail becomes critical as denoising progresses.

### A note on late-timestep anomalies

The **SD3** loss curves look roughly as one would expect from a standard flow matching model, though some compressed curves dip slightly below the baseline towards the end. One possible explanation is that SD3 has been fine-tuned with direct preference optimization, which can distort the learned velocity field away from the pure flow matching objective.

For **FLUX** and **AuraFlow**, the anomalies are more pronounced: the baseline loss curve itself has distinctive dents where it rises significantly above several compressed curves. FLUX is a distilled model, which may partially explain this, but the behavior is still unexpected. That said, these anomalies are confined to the later timesteps and do not affect the central hypothesis, which concerns the early steps where all models show clean, well-behaved loss curves.

## Project Structure

```
config.py          # Experiment hyperparameters (image size, compression levels, timesteps)
models.py          # Model loading and spatial compression utilities
experiments.py     # Main experiment loop — noise injection, model inference, MSE computation
main.ipynb         # Entry point — runs experiments and generates plots
```

## Models

| Model | Source |
|-------|--------|
| Stable Diffusion 3 | `stabilityai/stable-diffusion-3-medium-diffusers` |
| FLUX.1 | `black-forest-labs/FLUX.1-dev` |
| AuraFlow | `fal/AuraFlow-v0.3` |

## Requirements

- PyTorch
- `diffusers`, `transformers`
- `datasets` (HuggingFace)
- `plotly`, `scipy`, `numpy`
- A HuggingFace API token with access to the above models
