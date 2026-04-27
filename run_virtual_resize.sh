#!/usr/bin/env bash
# Equivalent to the original virtual_resize_experiment.py defaults.
python run_experiments.py --virtual-resize \
  --vr-models dit \
  --output all_experiment_results.pkl \
  "$@"
