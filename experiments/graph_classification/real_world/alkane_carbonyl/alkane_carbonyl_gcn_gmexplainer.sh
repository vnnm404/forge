#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset AlkaneCarbonyl \
    --model GCN \
    --in_dim 14 \
    --hidden_dim 64 \
    --out_dim 1 \
    --explanation_algorithm GraphMaskExplainer \
    --graph_epochs 50 \
    --complex_epochs 50 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations/ \
    --num_explanations 100 \
    --prop_strategy direct_prop \
    --alpha_c 0.5 \
    --alpha_e 0.5 \
    --start_seed 0 \
    --end_seed 1