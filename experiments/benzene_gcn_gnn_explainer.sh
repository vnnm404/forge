#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Benzene \
    --model GCN \
    --in_dim 14 \
    --hidden_dim 64 \
    --out_dim 1 \
    --explanation_algorithm GNNExplainer \
    --graph_epochs 100 \
    --complex_epochs 50 \
    --explanation_epochs 400 \
    --save_explanation_dir explanations/ \
    --num_explanations 100 \