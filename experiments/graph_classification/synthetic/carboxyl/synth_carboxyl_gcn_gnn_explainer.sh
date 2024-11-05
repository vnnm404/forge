#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Synth_carboxyl \
    --model GCN \
    --synth_shape_1 carboxyl \
    --synth_shape_2 no_carboxyl \
    --in_dim 8 \
    --hidden_dim 64 \
    --out_dim 1 \
    --explanation_algorithm PGMExplainer \
    --graph_epochs 50 \
    --complex_epochs 50 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 100 \
    --prop_strategy direct_prop \
    --alpha_c 0.0 \
    --alpha_e 0.0 \
    --start_seed 0 \
    --end_seed 1