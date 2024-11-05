#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Synth_carboxyl_6_cycle \
    --model GCN \
    --synth_shape_1 carboxyl \
    --synth_shape_2 cycle_6 \
    --in_dim 8 \
    --hidden_dim 64 \
    --out_dim 1 \
    --explanation_algorithm GNNExplainer \
    --graph_epochs 50 \
    --complex_epochs 50 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 100 \
    --prop_strategy hp_tuning \
    --alpha_c 0.0 \
    --alpha_e 0.5 \
    --start_seed 0 \
    --end_seed 10