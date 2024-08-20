#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Synth_bull_4_cycle \
    --model GCN \
    --synth_shape_1 bull \
    --synth_shape_2 cycle_4 \
    --in_dim 16 \
    --hidden_dim 64 \
    --out_dim 1 \
    --model_lr 0.5 \
    --explanation_algorithm GNNExplainer \
    --graph_epochs 50 \
    --complex_epochs 50 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 100 \
    --prop_strategy activation_prop \
    --alpha_c 1.5 \
    --alpha_e 1.5 \
    --start_seed 0 \
    --end_seed 2