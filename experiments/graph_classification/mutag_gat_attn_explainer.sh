#!/bin/bash

# Run the Mutag experiment with the GCN explainer
python3 main.py \
    --dataset Mutagenicity \
    --model GAT \
    --in_dim 14 \
    --hidden_dim 64 \
    --out_dim 1 \
    --explanation_algorithm AttentionExplainer \
    --graph_epochs 50 \
    --complex_epochs 100 \
    --explanation_epochs 400 \
    --save_explanation_dir explanations \
    --num_explanations 50 \
    --remove_type_2_nodes False \
    --spread_strategy cycle_wise
    # --test_graph_train_complex_dataset True \
    # --test_complex_train_graph_dataset True \