from data import load_dataset, get_data_loaders, load_dataset_as_complex
from models_graph import load_model
from eval_utils import train, test
from explain_utils import (
    initialise_explainer,
    explain_dataset,
    explanation_accuracy,
    visualise_explanation,
    save_to_graphml,
)
from config import device, args
import torch
import os
import json


def load_graph_data():
    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    train_loader, test_loader = get_data_loaders(dataset, batch_size=args.batch_size)
    print("Dataset loaded.")
    return dataset, train_loader, test_loader


def load_complex_data():
    print("Loading dataset...")
    complex_dataset = load_dataset_as_complex(args.dataset)
    train_loader, test_loader = get_data_loaders(complex_dataset, batch_size=64)
    print("Dataset loaded.")
    return complex_dataset, train_loader, test_loader


def setup_model(train_loader, test_loader, type="graphs"):
    model = load_model(
        name=args.model,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
    )
    model.to(device)
    model_path = os.path.join(args.save_dir, f"{args.exp_name}_{type}.pth")
    try:
        os.path.exists(model_path)
        print("Loading model...")
        model.load_state_dict(torch.load(model_path))
    except:
        print("Training model...")
        final_loss = train(
            model, train_loader, test_loader, model_path, epochs=args.graph_epochs
        )
        print(f"Final loss: {final_loss}")

    accuracy, precision, recall, f1 = test(model, test_loader)
    print(
        f"Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1: {f1}"
    )

    return model


def explain(model, dataset):
    explainer = initialise_explainer(
        model=model,
        explanation_algorithm_name=args.explanation_algorithm,
        explanation_epochs=args.explanation_epochs,
        explanation_lr=args.explanation_lr,
    )
    pred_explanations, ground_truth_explanations = explain_dataset(
        explainer, dataset, num=args.num_explanations
    )
    metrics = explanation_accuracy(ground_truth_explanations, pred_explanations)
    print(metrics)
    return pred_explanations, ground_truth_explanations, metrics


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


def save_graphml(dataset, explanation, type, is_gt=False):
    count = 0
    idx = 0
    while count < 10:
        try:
            save_to_graphml(
                data=dataset[idx][0],
                explanation=explanation[idx],
                outdir=args.save_explanation_graphml,
                fname=f"{args.exp_name}_{idx}_{type}.graphml",
                is_gt=is_gt,
            )
            count += 1
        except:
            pass
        idx += 1


def graph_classification():
    ########### GRAPH ############################
    dataset, train_loader, test_loader = load_graph_data()
    graph_model = setup_model(
        train_loader=train_loader, test_loader=test_loader, type="graphs"
    )
    graph_pred_explanations, ground_truth_explanations, metrics = explain(
        model=graph_model,
        dataset=dataset,
    )

    if args.visualise:
        visualise_explanation(graph_pred_explanations[1], ground_truth_explanations[1])

    if args.save_explanation_dir:
        save_metrics(metrics, args.exp_name, "graph")

    if args.save_explanation_graphml:
        save_graphml(dataset, graph_pred_explanations, "graph")
    
    if args.test_graph_train_complex_dataset:
        print("Testing explainer with model trained on graph, and providing complex dataset.")
        complex_dataset, _, _ = load_complex_data()
        explain(
            model=graph_model,
            dataset=dataset,
        )

    if args.visualise:
        visualise_explanation(graph_pred_explanations[1], ground_truth_explanations[1])

    if args.save_explanation_dir:
        save_metrics(metrics, args.exp_name, "graph")

    if args.save_explanation_graphml:
        save_graphml(dataset, graph_pred_explanations, "graph")

    ######### CELL COMPLEX ##########################
    complex_dataset, train_loader, test_loader = load_complex_data()
    model = setup_model(
        train_loader=train_loader, test_loader=test_loader, type="complexes"
    )
    complex_pred_explanations, _, complex_metrics = explain(
        model=model, dataset=complex_dataset
    )
    
    if args.test_complex_train_graph_dataset:
        print("Testing explainer with model trained on complexes, and providing graph dataset.")
        explain(
            model=model,
            dataset=dataset,
        )

    if args.visualise:
        visualise_explanation(
            complex_pred_explanations[1], ground_truth_explanations[1]
        )

    if args.save_explanation_dir:
        save_metrics(complex_metrics, args.exp_name, "complexes")

    if args.save_explanation_graphml:
        save_graphml(dataset, complex_pred_explanations, "complexes")

    ######### SAVE GROUND TRUTH ##########################
    if args.save_explanation_graphml:
        save_graphml(dataset, ground_truth_explanations, "gt", is_gt=True)
