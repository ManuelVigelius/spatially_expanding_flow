#!/usr/bin/env bash
# Equivalent to the original resize_experiment.py defaults.
python run_experiments.py --resize \
  --model dit \
  --dataset imagenet1k \
  --batch-size 32 \
  --cfg-scale 6.0 \
  --image-sizes $(seq 16 16 512) \
  --output resize_experiment_dit_imagenet1k_scaled_sigma.pkl \
  "$@"
