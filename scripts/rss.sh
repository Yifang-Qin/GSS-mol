#!/bin/bash
# GSS-Mol RSS baseline: random dihedral perturbation + relaxation

PROJECT_ROOT=$(pwd)
source $PROJECT_ROOT/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

# --- Configuration ---
calculator="ANI"          # ANI | UMA | UFF | MatterSim
num_samples=2048
output_dir="${PROJECT_ROOT}/sample/rss"
plot_dir="${PROJECT_ROOT}/plots/rss"

# --- RSS ---
gss-mol-rss \
    --output-dir $output_dir \
    --plot-dir $plot_dir \
    --calculator $calculator \
    --num-samples $num_samples

echo "Done. Results: $output_dir | Plots: $plot_dir"
