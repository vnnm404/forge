#!/bin/bash

# Run the benzene experiment with the GCN explainer
python3 main.py \
    --dataset Benzene \
    --model GCN \
    --in_dim 14 \
    --hidden_dim 64 \
    --out_dim 1 \
    --expl_type node \
    --explanation_algorithm SubgraphX \
    --explanation_aggregation threshold \
    --graph_epochs 50 \
    --complex_epochs 50 \
    --explanation_epochs 100 \
    --save_explanation_dir explanations \
    --num_explanations 100 \
    --prop_strategy direct_prop \
    --alpha_c 3.0 \
    --alpha_e 0.0 \
    --start_seed 0 \
    --end_seed 10
    # --prop_strategy direct_prop
    # --remove_type_1_nodes \