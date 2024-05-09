#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --task_level node \
    --dataset Triangle \
    --model GCN \
    --in_dim 14 \
    --hidden_dim 16 \
    --out_dim 1 \
    --explanation_algorithm GNNExplainer \
    --graph_epochs 50 \
    --complex_epochs 1000 \
    --explanation_epochs 400 \
    --save_explanation_dir explanations \
    --num_explanations 50 \
    --spread_strategy cycle_wise \
    --start_seed 0 \
    --end_seed 10