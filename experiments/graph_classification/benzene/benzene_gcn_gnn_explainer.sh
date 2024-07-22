#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Benzene \
    --model GCN \
    --in_dim 14 \
    --hidden_dim 64 \
    --out_dim 1 \
    --explanation_algorithm GNNExplainer \
    --graph_epochs 50 \
    --complex_epochs 50 \
    --explanation_epochs 200 \
    --save_explanation_dir explanations \
    --num_explanations 100 \
    --spread_strategy cycle_wise\
    --alpha_c 1.5 \
    --alpha_e 0.5 \
    --start_seed 0 \
    --end_seed 1
    # --spread_strategy cycle_wise
    # --remove_type_1_nodes \