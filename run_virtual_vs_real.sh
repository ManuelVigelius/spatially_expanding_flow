#!/usr/bin/env bash
# Equivalent to the original virtual_vs_real_resize_experiment.py defaults.
python run_experiments.py --virtual-vs-real \
  --full-size 1024 \
  --vvr-batch-size 16 \
  --vvr-sizes 64 128 256 512 768 1024 \
  --vvr-dataset div2k \
  --output virtual_vs_real_resize_experiment_results_1024.pkl \
  "$@"
