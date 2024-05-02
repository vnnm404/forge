#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Benzene \
    --model GAT \
    --in_dim 14 \
    --hidden_dim 64 \
    --out_dim 1 \
    --explanation_algorithm AttentionExplainer \
    --graph_epochs 50 \
    --complex_epochs 50 \
    --explanation_epochs 400 \
    --save_explanation_dir explanations \
    --num_explanations 50 \
    --remove_type_2_nodes False \
    --test_graph_train_complex_dataset True \
    --test_complex_train_graph_dataset True