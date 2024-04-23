import torch
from graphxai.datasets import ShapeGGen
from models_node import load_model
from graphxai.gnn_models.node_classification import train, test
from config import args
from explain_utils import initialise_explainer, explain_nodes, explanation_accuracy
from data import graph_to_complex
import os
import json

# dataset = torch.load("data/ShapeGGen.pt")
# print(dataset)

def load_data():
    dataset = ShapeGGen(
        model_layers = 2,
        num_subgraphs = 15,
        subgraph_size = 12,
        prob_connection = 1,
        add_sensitive_feature = False,
        seed=1234
    )
    data = dataset.get_graph(use_fixed_split=True)

    return dataset, data

def load_data_as_complex():
    dataset = ShapeGGen(
        model_layers = 2,
        num_subgraphs = 20,
        subgraph_size = 15,
        prob_connection = 0.3,
        add_sensitive_feature = False
    )
    data = dataset.get_graph(use_fixed_split=True)
    og_num_nodes = data.x.shape[0]
    complex_data, mapping = graph_to_complex(data)
    complex_data = complex_data.to_homogeneous()
    final_num_nodes = complex_data.x.shape[0]
    
    extension = final_num_nodes - og_num_nodes
    
    # pad the y, train mask, val mask, test mask
    complex_data.y = torch.cat([complex_data.y, torch.zeros(extension, dtype = torch.long)])
    complex_data.train_mask = torch.cat([data.train_mask, torch.zeros(extension, dtype = torch.bool)])
    complex_data.val_mask = torch.cat([data.valid_mask, torch.zeros(extension, dtype = torch.bool)])
    complex_data.test_mask = torch.cat([data.test_mask, torch.zeros(extension, dtype = torch.bool)])
    
    return dataset, complex_data, mapping

def setup_model(data, in_features):
    model = load_model(
        name = args.model,
        in_dim = in_features,
        hidden_dim = args.hidden_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
    criterion = torch.nn.CrossEntropyLoss()
    # Train model:
    for _ in range(1000):
        loss = train(model, optimizer, criterion, data)

    f1, acc, prec, rec, auprc, auroc = test(model, data, num_classes = 2, get_auc = True)

    print(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\nAUPRC: {auprc}\nAUROC: {auroc}")

    return model

def explain(model, data, dataset, mapping=None, type="g"):
    explainer = initialise_explainer(
        model=model,
        explanation_algorithm_name=args.explanation_algorithm,
        explanation_epochs=args.explanation_epochs,
        explanation_lr=args.explanation_lr,
        task="multiclass_classification"
    )
    pred_explanations, gt_explanations = explain_nodes(explainer, data, dataset, mapping, type=type)

    metrics = explanation_accuracy(gt_explanations, pred_explanations)
    print(metrics)
    return pred_explanations, gt_explanations, metrics

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

def node_classification():
    ######### Graphs ################
    dataset, data = load_data()
    model = setup_model(data, dataset.n_features)
    pred_explanations, gt_explanations, metrics = explain(model, data, dataset)
    
    if args.save_explanation_dir:
        save_metrics(metrics, args.exp_name, "graph")

    ######### Complex ################
    dataset, complex_data, mapping = load_data_as_complex()
    model = setup_model(complex_data, dataset.n_features)
    pred_explanations, gt_explanations, metrics = explain(model, complex_data, dataset, mapping, type="c")
    
    if args.save_explanation_dir:
        save_metrics(metrics, args.exp_name, "complex")