#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Mutagenicity \
    --model GCN \
    --in_dim 14 \
    --hidden_dim 64 \
    --out_dim 1 \
    --explanation_algorithm GNNExplainer \
    --graph_epochs 10 \
    --complex_epochs 400 \
    --model_lr 0.5 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 500 \
    --remove_type_2_nodes False \
    --spread_strategy edge_wise \
    --start_seed 205 \
    --end_seed 210 \