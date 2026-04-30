#!/bin/bash
# GSS-Mol guided sampling + visualization pipeline

PROJECT_ROOT=$(pwd)
source $PROJECT_ROOT/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

# --- Configuration ---
config="config/sample.yaml"
sample_name="gss_sample"
calculator="ANI"          # ANI | UMA | UFF | MatterSim
batch_size=1024
sample_num=4096

out_dir="${PROJECT_ROOT}/sample/${sample_name}"
plot_dir="${PROJECT_ROOT}/plots/${sample_name}"
mkdir -p ${out_dir} ${plot_dir}

out_path="${out_dir}/${calculator}_guided.xyz"
plot_path="${plot_dir}/${calculator}_guided.png"

# --- Guided sampling ---
gss-mol-sample --conf $config \
    --calculator $calculator \
    --batch-size $batch_size \
    --sample-num $sample_num \
    --output $out_path

# --- Visualization ---
gss-mol-plot --input $out_path --output $plot_path

echo "Done. Samples: $out_path | Plot: $plot_path"
