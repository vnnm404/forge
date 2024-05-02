#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --task_level node \
    --dataset Diamond \
    --model GAT \
    --in_dim 14 \
    --hidden_dim 64 \
    --out_dim 1 \
    --explanation_algorithm GraphMaskExplainer \
    --graph_epochs 50 \
    --complex_epochs 100 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 50 \
    --save_explanation_graphml graphml \
    --remove_type_2_nodes False \