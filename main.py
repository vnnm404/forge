from data import load_dataset, get_data_loaders, load_dataset_as_complex
from models import GCN
from eval_utils import train, test
from explain_utils import (
    initialise_explainer,
    explain_dataset,
    explanation_accuracy,
    visualise_explanation,
    save_to_graphml,
)
from config import device
import torch
import os

if __name__ == "__main__":
    ##### DATA LOAD AND PREPROCESSING #####
    print("Loading dataset...")
    dataset = load_dataset("Benzene")
    train_loader, test_loader = get_data_loaders(dataset, batch_size=64)
    print("Dataset loaded.")
    ##### MODEL #####
    model = GCN(
        in_dim=14, hidden_dim=64, out_dim=1
    )  # TODO: put everything in config + argparse
    model.to(device)
    ##### TRAIN/LOAD #####
    # if available, load model, else train model
    model_path = "trained_models/graph_gcn.pth"
    if os.path.exists(model_path):
        print("Loading model...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training model...")
        final_loss = train(model, train_loader, epochs=100)
        print(f"Final loss: {final_loss}")
        # save model
        torch.save(model.state_dict(), model_path)

    ##### TEST #####
    accuracy, precision, recall, f1 = test(model, test_loader)
    print(
        f"Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1: {f1}"
    )
    ##### EXPLANATION #####
    explainer = initialise_explainer(model, explanation_algorithm_name="GNNExplainer")

    pred_explanations, ground_truth_explanations = explain_dataset(explainer, dataset)

    metrics = explanation_accuracy(ground_truth_explanations, pred_explanations)

    print(f"Explanation accuracy: {metrics['accuracy']}")
    print(f"Explanation precision: {metrics['precision']}")
    print(f"Explanation recall: {metrics['recall']}")
    print(f"Explanation f1: {metrics['f1']}")

    # visualise the first explanation
    visualise_explanation(pred_explanations[1], ground_truth_explanations[1])

    # save the first explanation to graphml
    save_to_graphml(dataset[2][0], pred_explanations[1], "graph_explanation.graphml")

    ######### CELL COMPLEX ##########################

    ##### DATA LOAD AND PREPROCESSING #####
    print("Loading dataset...")
    complex_dataset = load_dataset_as_complex("Benzene")
    train_loader, test_loader = get_data_loaders(complex_dataset, batch_size=64)

    ##### MODEL #####
    model = GCN(
        in_dim=14, hidden_dim=64, out_dim=1
    )  # TODO: put everything in config + argparse
    model.to(device)
    model_path = "trained_models/complex_gcn.pth"
    ##### TRAIN/LOAD #####
    if os.path.exists(model_path):
        print("Loading model...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training model...")
        final_loss = train(model, train_loader, epochs=10)
        print(f"Final loss: {final_loss}")
        # save model
        torch.save(model.state_dict(), model_path)

    ##### TEST #####
    accuracy, precision, recall, f1 = test(model, test_loader)
    print(
        f"Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1: {f1}"
    )

    ##### EXPLANATION #####
    explainer = initialise_explainer(model, explanation_algorithm_name="GNNExplainer")

    pred_explanations, ground_truth_explanations = explain_dataset(
        explainer, complex_dataset
    )

    metrics = explanation_accuracy(ground_truth_explanations, pred_explanations)

    print(f"Explanation accuracy: {metrics['accuracy']}")
    print(f"Explanation precision: {metrics['precision']}")
    print(f"Explanation recall: {metrics['recall']}")
    print(f"Explanation f1: {metrics['f1']}")

    # visualise the first explanation
    visualise_explanation(pred_explanations[1], ground_truth_explanations[1])

    # save the first explanation to graphml
    save_to_graphml(dataset[2][0], pred_explanations[1], "complex_explanation.graphml")

# save ground truth explanation to graphml
save_to_graphml(
    dataset[2][0],
    ground_truth_explanations[1],
    "ground_truth_explanation.graphml",
    is_gt=True,
)
