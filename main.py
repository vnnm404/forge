from data import load_dataset, get_data_loaders, load_dataset_as_complex
from models import load_model
from eval_utils import train, test
from explain_utils import (
    initialise_explainer,
    explain_dataset,
    explanation_accuracy,
    visualise_explanation,
    save_to_graphml,
)
from config import device, args
from time import time
import torch
import os
import json


if __name__ == "__main__":
    # ##### DATA LOAD AND PREPROCESSING #####
    # print("Loading dataset...")
    # dataset = load_dataset(args.dataset)
    # train_loader, test_loader = get_data_loaders(dataset, batch_size=args.batch_size)
    # print("Dataset loaded.")

    # ##### MODEL #####
    # model = load_model(
    #     name=args.model,
    #     in_dim=args.in_dim,
    #     hidden_dim=args.hidden_dim,
    #     out_dim=args.out_dim,
    # )
    # model.to(device)

    # ##### TRAIN/LOAD #####
    # model_path = os.path.join(args.save_dir, f"{args.exp_name}_graphs.pth")
    # try:
    #     os.path.exists(model_path)
    #     print("Loading model...")
    #     model.load_state_dict(torch.load(model_path))
    # except:
    #     print("Training model...")
    #     final_loss = train(model, train_loader, test_loader, model_path, epochs=args.graph_epochs)
    #     print(f"Final loss: {final_loss}")
    #     # save model
    #     # torch.save(model.state_dict(), model_path)

    # ##### TEST #####
    # accuracy, precision, recall, f1 = test(model, test_loader)
    # print(
    #     f"Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1: {f1}"
    # )

    # ##### EXPLANATION #####
    # explainer = initialise_explainer(
    #     model=model,
    #     explanation_algorithm_name=args.explanation_algorithm,
    #     explanation_epochs=args.explanation_epochs,
    #     explanation_lr=args.explanation_lr,
    # )
    
    # if args.explanation_algorithm == "PGExplainer":
    #     # PGExplainer needs to be trained first
    #     print("Training PGExplainer...")
    #     for epoch in range(args.explanation_epochs):
    #         for i in range(args.num_explanations):
    #             data = dataset[i][0]
    #             explainer.algorithm.train(
    #                 epoch=epoch,
    #                 model=model,
    #                 x=data.x,
    #                 edge_index=data.edge_index,
    #                 batch=data.batch,
    #             )
    #     print("PGExplainer trained.")
    # pred_explanations, ground_truth_explanations = explain_dataset(
    #     explainer, dataset, num=args.num_explanations
    # )

    # print(len(pred_explanations), len(ground_truth_explanations))

    # metrics = explanation_accuracy(ground_truth_explanations, pred_explanations)

    # print(metrics)

    # if args.visualise:
    #     # visualise the first explanation
    #     visualise_explanation(pred_explanations[1], ground_truth_explanations[1])
    # if args.save_explanation_dir:
    #     # save metrics to json
    #     metrics_path = os.path.join(
    #         args.save_explanation_dir,
    #         f"{args.exp_name}",
    #         f"{time()}_graph_metrics.json",
    #     )
    #     # create directory if it doesn't exist
    #     os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    #     with open(metrics_path, "w") as f:
    #         json.dump(metrics, f, indent=4)

    # if args.save_explanation_graphml:
    #     # save the first explanation to graphml
    #     save_to_graphml(
    #         data=dataset[1][0],
    #         explanation=pred_explanations[1],
    #         outdir=args.save_explanation_graphml,
    #         fname=f"{args.exp_name}_graph.save_to_graphml",
    #     )
    ######### CELL COMPLEX ##########################

    ##### DATA LOAD AND PREPROCESSING #####
    print("Loading dataset...")
    complex_dataset = load_dataset_as_complex("Benzene")
    train_loader, test_loader = get_data_loaders(complex_dataset, batch_size=64)

    ##### MODEL #####
    model = load_model(
        args.model,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
    )
    model.to(device)
    model_path = os.path.join(args.save_dir, f"{args.exp_name}_complexes.pth")
    ##### TRAIN/LOAD #####
    try:
        os.path.exists(model_path)
        print("Loading model...")
        model.load_state_dict(torch.load(model_path))
    except:
        print("Training model...")
        final_loss = train(model, train_loader, test_loader, model_path, epochs=args.complex_epochs)
        print(f"Final loss: {final_loss}")
        # save model
        # torch.save(model.state_dict(), model_path)

    ##### TEST #####
    accuracy, precision, recall, f1 = test(model, test_loader)
    print(
        f"Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1: {f1}"
    )

    ##### EXPLANATION #####
    explainer = initialise_explainer(
        model=model,
        explanation_algorithm_name=args.explanation_algorithm,
        explanation_epochs=args.explanation_epochs,
        explanation_lr=args.explanation_lr,
    )

    if args.explanation_algorithm == "PGExplainer":
        # PGExplainer needs to be trained first
        print("Training PGExplainer...")
        for epoch in range(args.explanation_epochs):
            for i in range(args.num_explanations):
                data = complex_dataset[i][0]
                explainer.algorithm.train(
                    epoch=epoch,
                    model=model,
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch,
                )

    pred_explanations, ground_truth_explanations = explain_dataset(
        explainer, complex_dataset, num=args.num_explanations
    )

    metrics = explanation_accuracy(ground_truth_explanations, pred_explanations)

    print(metrics)

    if args.visualise:
        # visualise the first explanation
        visualise_explanation(pred_explanations[1], ground_truth_explanations[1])
    if args.save_explanation_dir:
        # save metrics to json
        metrics_path = os.path.join(
            args.save_explanation_dir,
            f"{args.exp_name}",
            f"{time()}_complex_metrics.json",
        )
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    if args.save_explanation_graphml:
        # save the first explanation to graphml
        save_to_graphml(
            data=dataset[1][0],
            explanation=pred_explanations[1],
            outdir=args.save_explanation_graphml,
            fname=f"{args.exp_name}_complexes.graphml",
        )

    if args.save_explanation_graphml:
        # save ground truth explanation to graphml
        save_to_graphml(
            data=dataset[2][0],
            explanation=ground_truth_explanations[1],
            outdir=args.save_explanation_graphml,
            fname=f"{args.exp_name}_gt.graphml",
            is_gt=True,
        )
