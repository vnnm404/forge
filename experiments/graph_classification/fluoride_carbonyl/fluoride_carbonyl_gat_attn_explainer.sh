#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset FluorideCarbonyl \
    --model GAT \
    --in_dim 14 \
    --hidden_dim 32 \
    --out_dim 1 \
    --explanation_algorithm AttentionExplainer \
    --graph_epochs 50 \
    --complex_epochs 30 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations/ \
    --num_explanations 1125 \
    --remove_type_2_nodes True \
    --spread_strategy edge_wise \
    --start_seed 0 \
    --end_seed 10