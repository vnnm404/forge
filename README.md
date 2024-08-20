Code for AAAI submission: Higher Order Structures in Graph Explanations

The codebase is built with `Python 3.10.14`.

To create a new conda environment with the required python version, use the following command:
```
conda create -n forge python=3.10.14
```

Please use the below command to install the required packages in their specific versions:
```
pip install -r requirements.txt
```

## Running the code
Multiple scripts are provided to run the experiments. The scripts are located in the `experiments` directory. There are both real-world and synthetic datasets present in the `real_world` and `synthetic` directories respectively. The scripts are named as `<dataset>_<gnn>_<explainer>.sh`.

For example, to run the experiments on the `benzene` dataset using the `gcn` model and the `gnn_explainer` explainer, run the following command:
```
sh experiments/graph_classification/real_world/benzene/benzene_gcn_gnn_explainer.sh
```

The results will be saved in the `explanations` directory, under appropriate sub-directories. The results are saved in json files, with separate json files for baseline and FORGE results (`graphs.json` and `complexes.json` respectively). containing per-seed as well as average results.
