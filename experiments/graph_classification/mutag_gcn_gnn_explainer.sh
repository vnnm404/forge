#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Mutagenicity \
    --model GCN \
    --in_dim 14 \
    --hidden_dim 32 \
    --out_dim 1 \
    --explanation_algorithm GNNExplainer \
    --graph_epochs 50 \
    --complex_epochs 50 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 500 \
    --remove_type_2_nodes True \
    --spread_strategy cycle_wise \
    --test_graph_train_complex_dataset True \
    --start_seed 0 \
    --end_seed 10