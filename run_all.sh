#!/usr/bin/env bash
# Reproduce every table and figure from the paper.
set -e
cd "$(dirname "$0")"
python run_experiment.py --dataset nasa
python run_experiment.py --dataset wc98 --autoscaler
echo
echo "Done. See results/tables/*.csv and results/figures/*.png"
