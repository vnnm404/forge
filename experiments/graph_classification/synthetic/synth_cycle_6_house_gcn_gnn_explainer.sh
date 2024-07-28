#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Synth_cycle_6_house \
    --model GCN \
    --synth_shape_1 cycle_6 \
    --synth_shape_2 house \
    --in_dim 16 \
    --hidden_dim 64 \
    --out_dim 1 \
    --model_lr 0.1 \
    --explanation_algorithm GNNExplainer \
    --graph_epochs 50 \
    --complex_epochs 50 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 100 \
    --prop_strategy hierarchical_prop \
    --alpha_c 1.0 \
    --alpha_e 1.0 \
    --start_seed 0 \
    --end_seed 1