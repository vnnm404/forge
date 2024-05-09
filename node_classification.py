from pprint import pprint
import torch
from tqdm import tqdm
from graphxai.datasets import ShapeGGen
from models_node import load_model
from graphxai.gnn_models.node_classification import train, test
from config import args
from explain_utils import (
    initialise_explainer,
    explain_nodes,
    explanation_accuracy,
    save_node_graph_to_graphml,
)
from data import graph_to_complex
import os
import json
from graphxai.datasets.utils.shapes import (
    house,
    diamond,
    wheel,
    triangle,
)


# dataset = torch.load("data/ShapeGGen.pt")
# print(dataset)
def shape():
    if args.dataset == "House":
        return house
    elif args.dataset == "Diamond":
        return diamond
    elif args.dataset == "Wheel":
        return wheel
    elif args.dataset == "Triangle":
        return triangle
    else:
        return wheel

def load_data(seed):
    dataset = ShapeGGen(
        shape=shape(),
        model_layers=2,
        num_subgraphs=15,
        subgraph_size=12,
        prob_connection=1,
        add_sensitive_feature=False,
        seed=seed,
    )
    data = dataset.get_graph(use_fixed_split=True)
    return dataset, data


def load_data_as_complex(seed):
    dataset = ShapeGGen(
        shape=shape(),
        model_layers=2,
        num_subgraphs=15,
        subgraph_size=12,
        prob_connection=1,
        add_sensitive_feature=False,
        seed=seed,
    )
    data = dataset.get_graph(use_fixed_split=True)
    og_num_nodes = data.x.shape[0]
    complex_data, mapping = graph_to_complex(data)
    complex_data = complex_data.to_homogeneous()
    final_num_nodes = complex_data.x.shape[0]
    extension = final_num_nodes - og_num_nodes

    # pad the y, train mask, val mask, test mask
    complex_data.y = torch.cat(
        [complex_data.y, torch.zeros(extension, dtype=torch.long)]
    )
    complex_data.train_mask = torch.cat(
        [data.train_mask, torch.zeros(extension, dtype=torch.bool)]
    )
    complex_data.val_mask = torch.cat(
        [data.valid_mask, torch.zeros(extension, dtype=torch.bool)]
    )
    complex_data.test_mask = torch.cat(
        [data.test_mask, torch.zeros(extension, dtype=torch.bool)]
    )

    return dataset, complex_data, mapping


def setup_model(data, in_features):
    model = load_model(
        name=args.model,
        in_dim=in_features,
        hidden_dim=args.hidden_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    # Train model:
    for _ in range(1000):
        loss = train(model, optimizer, criterion, data)

    f1, acc, prec, rec, auprc, auroc = test(model, data, num_classes=2, get_auc=True)

    print(
        f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\nAUPRC: {auprc}\nAUROC: {auroc}"
    )

    return model


def explain(model, data, dataset, mapping=None, type="g"):
    explainer = initialise_explainer(
        model=model,
        explanation_algorithm_name=args.explanation_algorithm,
        explanation_epochs=args.explanation_epochs,
        explanation_lr=args.explanation_lr,
        task="multiclass_classification",
    )
    pred_explanations, gt_explanations, edge_idxs = explain_nodes(
        explainer, data, dataset, mapping, type=type
    )

    metrics = explanation_accuracy(gt_explanations, pred_explanations)
    print(metrics)
    return pred_explanations, gt_explanations, edge_idxs, metrics


def save_metrics(metrics, exp_name, type):
    metrics_path = os.path.join(
        args.save_explanation_dir,
        f"{exp_name}",
        f"{args.time}",
        f"{type}_metrics.json",
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_graphml(edge_idxs, explanation, type, is_gt=False):
    count = 0
    idx = 0
    while count < 10 and idx < len(edge_idxs):
        try:
            save_node_graph_to_graphml(
                edge_index=edge_idxs[idx],
                explanation=explanation[idx],
                node_idx=idx,
                outdir=args.save_explanation_graphml,
                fname=f"{args.exp_name}/{idx}/{type}.graphml",
                is_gt=is_gt,
            )
            count += 1
        except Exception as e:
            pass
        idx += 1


def node_classification():
    ######### Graphs ################
    print("Running graph setup")
    graph_metrics = {}
    for seed in tqdm(range(args.start_seed, args.end_seed)):
        args.current_seed = seed
        dataset, data = load_data(seed=seed)
        model = setup_model(data, dataset.n_features)
        pred_explanations, gt_explanations, graph_edge_idxs, metrics = explain(
            model, data, dataset
        )

        graph_metrics[seed] = metrics

        if args.save_explanation_graphml:
            save_graphml(
                graph_edge_idxs,
                pred_explanations,
                "graph",
            )

    best_seed = max(graph_metrics, key=graph_metrics.get("jaccard"))
    best_metrics = graph_metrics[best_seed]
    print(f"Best seed for graph explanations: {best_seed}")
    print(f"Best metrics for graph explanations: ")
    pprint(best_metrics)

    if args.save_explanation_dir:
        # sort metrics by jaccard score
        graph_metrics = dict(
            sorted(graph_metrics.items(), key=lambda x: x[1]["jaccard"], reverse=True)
        )
        save_metrics(graph_metrics, args.exp_name, "graph")

    ######### Complex ################
    complex_metrics = {}
    print("Running complex setup")

    for seed in tqdm(range(args.start_seed, args.end_seed)):
        args.current_seed = seed
        dataset, complex_data, mapping = load_data_as_complex(seed=seed)
        model = setup_model(complex_data, dataset.n_features)
        pred_explanations, gt_explanations, complex_edge_idxs, metrics = explain(
            model, complex_data, dataset, mapping, type="c"
        )

        complex_metrics[seed] = metrics

        if args.save_explanation_graphml:
            save_graphml(
                complex_edge_idxs,
                pred_explanations,
                "complex",
            )

        ######### GROUND TRUTH ################
        if args.save_explanation_graphml:
            save_graphml(complex_edge_idxs, gt_explanations, "ground_truth", is_gt=True)

    # get best seed based on jaccard score
    best_seed = max(complex_metrics, key=complex_metrics.get("jaccard"))
    best_metrics = complex_metrics[best_seed]
    print(f"Best seed for complex explanations: {best_seed}")
    print(f"Best metrics for complex explanations: ")
    pprint(best_metrics)

    if args.save_explanation_dir:
        # sort metrics by jaccard score
        complex_metrics = dict(
            sorted(complex_metrics.items(), key=lambda x: x[1]["jaccard"], reverse=True)
        )
        save_metrics(complex_metrics, args.exp_name, "complexes")
