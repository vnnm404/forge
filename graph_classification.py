from tqdm import tqdm
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
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_graph_data(seed):
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, seed=seed)
    train_loader, test_loader = get_data_loaders(dataset, batch_size=args.batch_size)
    # get number of samples in each class for train and test loader
    print("Train loader class distribution:")
    for i in range(2):
        print(
            f"Class {i}: {len([data for data in train_loader.dataset if data.y == i])}"
        )
    print("Test loader class distribution:")
    for i in range(2):
        print(
            f"Class {i}: {len([data for data in test_loader.dataset if data.y == i])}"
        )
    print("Dataset loaded.")
    return dataset, train_loader, test_loader


def load_complex_data(seed):
    print("Loading dataset...")
    complex_dataset = load_dataset_as_complex(args.dataset, seed=seed)
    train_loader, test_loader = get_data_loaders(complex_dataset, batch_size=64)
    # get number of samples in each class for train and test loader
    print("Train loader class distribution:")
    for i in range(2):
        print(
            f"Class {i}: {len([data for data in train_loader.dataset if data.y == i])}"
        )
    print("Test loader class distribution:")
    for i in range(2):
        print(
            f"Class {i}: {len([data for data in test_loader.dataset if data.y == i])}"
        )
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
    print(device)
    model_path = os.path.join(
        args.save_dir,
        f"{args.exp_name}",
        f"{args.current_seed}",
        f"{args.model}_{type}.pth",
    )
    try:
        os.path.exists(model_path)
        print("Loading model...")
        model.load_state_dict(torch.load(model_path))
    except:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print("Training model...")
        final_loss = train(
            model,
            train_loader,
            test_loader,
            model_path,
            epochs=args.graph_epochs if type == "graphs" else args.complex_epochs,
        )
        print(f"Final loss: {final_loss}")

    accuracy, precision, recall, f1 = test(model, test_loader)
    print(
        f"Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1: {f1}"
    )

    return model


def explain(model, dataset, graph_explainer=None):
    explainer = initialise_explainer(
        model=model,
        explanation_algorithm_name=args.explanation_algorithm,
        explanation_epochs=args.explanation_epochs,
        explanation_lr=args.explanation_lr,
    )
    pred_explanations, ground_truth_explanations, faithfulness = explain_dataset(
        explainer, dataset, num=args.num_explanations, graph_explainer=graph_explainer
    )
    metrics = explanation_accuracy(ground_truth_explanations, pred_explanations)
    metrics["faithfulness"] = faithfulness
    print(metrics)
    return pred_explanations, ground_truth_explanations, metrics, explainer


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
    # print("Running graph setup")
    graph_metrics = {}
    explainers = {}
    # for seed in tqdm(range(args.start_seed, args.end_seed), desc="Graph Setup Seed: "):
    #     args.current_seed = seed

    #     dataset, train_loader, test_loader = load_graph_data(seed=seed)

    #     graph_model = setup_model(
    #         train_loader=train_loader, test_loader=test_loader, type="graphs"
    #     )
    #     graph_pred_explanations, ground_truth_explanations, metrics, explainer = explain(
    #         model=graph_model,
    #         dataset=dataset,
    #     )
        
    #     explainers[seed] = explainer

    #     if args.visualise:
    #         visualise_explanation(
    #             graph_pred_explanations[1], ground_truth_explanations[1]
    #         )

    #     graph_metrics[seed] = metrics

    #     if args.save_explanation_graphml:
    #         save_graphml(dataset, graph_pred_explanations, "graph")

    #     if args.test_graph_train_complex_dataset:
    #         print(
    #             "Testing explainer with model trained on graph, and providing complex dataset."
    #         )
    #         complex_dataset, _, _ = load_complex_data(seed=seed)
    #         explain(
    #             model=graph_model,
    #             dataset=dataset,
    #         )

    #     if args.visualise:
    #         visualise_explanation(
    #             graph_pred_explanations[1], ground_truth_explanations[1]
    #         )

    #     if args.save_explanation_graphml:
    #         save_graphml(dataset, graph_pred_explanations, "graph")

    # # get best seed based on jaccard score
    # best_seed = max(graph_metrics, key=graph_metrics.get("jaccard"))
    # best_metrics = graph_metrics[best_seed]
    # print(f"Best seed for graph explanations: {best_seed}")
    # print(f"Best metrics for graph explanations: ")
    # pprint(best_metrics)

    # if args.save_explanation_dir:
    #     # sort metrics by jaccard score
    #     graph_metrics = dict(
    #         sorted(graph_metrics.items(), key=lambda x: x[1]["jaccard"], reverse=True)
    #     )

    #     # average metrics
    #     avg_metrics = {}
    #     for key in graph_metrics[seed].keys():
    #         avg_metrics[key] = sum([x[key] for x in graph_metrics.values()]) / len(
    #             graph_metrics
    #         )

    #     graph_metrics["average"] = avg_metrics

    #     # std dev metrics
    #     std_metrics = {}
    #     for key in graph_metrics[seed].keys():
    #         std_metrics[key] = np.std([x[key] for x in graph_metrics.values()])

    #     graph_metrics["std_dev"] = std_metrics

    #     save_metrics(graph_metrics, args.exp_name, "graph")

    ######### CELL COMPLEX ##########################
    complex_metrics = {}
    print("Running complex setup")
    for seed in tqdm(range(args.start_seed, args.end_seed), desc="Complex Setup Seed:"):
        args.current_seed = seed

        complex_dataset, train_loader, test_loader = load_complex_data(seed=seed)

        model = setup_model(
            train_loader=train_loader, test_loader=test_loader, type="complexes"
        )
        complex_pred_explanations, _, metrics, _ = explain(
            model=model, dataset=complex_dataset, # graph_explainer=explainers[seed]
        )

        # if args.test_complex_train_graph_dataset:
        #     print(
        #         "Testing explainer with model trained on complexes, and providing graph dataset."
        #     )
        #     explain(
        #         model=model,
        #         dataset=dataset,
        #     )

        # if args.visualise:
        #     visualise_explanation(
        #         complex_pred_explanations[1], ground_truth_explanations[1]
        # )

        complex_metrics[seed] = metrics

        # if args.save_explanation_graphml:
        #     save_graphml(dataset, complex_pred_explanations, "complexes")

        ######### SAVE GROUND TRUTH ##########################
        # if args.save_explanation_graphml:
        #     save_graphml(dataset, ground_truth_explanations, "gt", is_gt=True)

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

        # average metrics
        avg_metrics = {}
        for key in complex_metrics[seed].keys():
            avg_metrics[key] = sum([x[key] for x in complex_metrics.values()]) / len(
                complex_metrics
            )

        complex_metrics["average"] = avg_metrics

        # std dev metrics
        std_metrics = {}
        for key in complex_metrics[seed].keys():
            std_metrics[key] = np.std([x[key] for x in complex_metrics.values()])

        complex_metrics["std_dev"] = std_metrics

        save_metrics(complex_metrics, args.exp_name, "complexes")
