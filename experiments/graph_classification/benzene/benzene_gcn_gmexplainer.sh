#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Benzene \
    --model GCN \
    --in_dim 14 \
    --hidden_dim 32 \
    --out_dim 1 \
    --explanation_algorithm GraphMaskExplainer \
    --graph_epochs 50 \
    --complex_epochs 30 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 150 \
    --remove_type_2_nodes False \
    --prop_strategy direct_prop\
    --start_seed 0 \
    --end_seed 10