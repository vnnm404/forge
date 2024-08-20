#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Mutagenicity \
    --model GCN \
    --in_dim 14 \
    --hidden_dim 64 \
    --out_dim 1 \
    --model_lr 0.05 \
    --explanation_algorithm GNNExplainer \
    --graph_epochs 50 \
    --complex_epochs 300 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 100 \
    --prop_strategy direct_prop \
    --alpha_c 0 \
    --alpha_e 0 \
    --start_seed 0 \
    --end_seed 1