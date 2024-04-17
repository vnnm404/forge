import torch
import argparse


def load_args():
    parser = argparse.ArgumentParser(
        description="HOGE: Higher Order Graph Explanations"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Benzene",
        help="The dataset to use",
        choices=["Benzene", "AlkaneCarbonyl"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        help="The model to use",
        choices=["GCN"],
    )
    parser.add_argument(
        "--in_dim",
        type=int,
        default=14,
        help="The input dimension of the model",
    )
    parser.add_argument(
        "--out_dim",
        type=int,
        default=1,
        help="The output dimension of the model",
    )
    parser.add_argument(
        "--model_lr",
        type=float,
        default=0.01,
        help="The learning rate of the model",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="The hidden dimension of the model",
    )
    parser.add_argument(
        "--explanation_algorithm",
        type=str,
        default="GNNExplainer",
        help="The explanation algorithm to use",
        choices=["GNNExplainer", "PGExplainer", "Captum"],
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The batch size for training"
    )
    parser.add_argument(
        "--graph_epochs",
        type=int,
        default=100,
        help="The number of epochs for training on graphs",
    )
    parser.add_argument(
        "--complex_epochs",
        type=int,
        default=100,
        help="The number of epochs for training on cell complexes",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="trained_models",
        help="The directory to save models",
    )
    parser.add_argument(
        "--explanation_epochs",
        type=int,
        default=200,
        help="The number of epochs for training the explanation algorithm",
    )
    parser.add_argument(
        "--explanation_lr",
        type=float,
        default=0.01,
        help="The learning rate for training the explanation algorithm",
    )
    parser.add_argument(
        "--visualise",
        type=bool,
        default=False,
        help="Whether to visualise the explanation",
    )
    parser.add_argument(
        "--save_explanation_dir",
        type=str,
        default=None,
        help="The directory to save explanations",
    )
    parser.add_argument(
        "--save_explanation_graphml",
        type=str,
        default=None,
        help="The directory to save explanations in graphml format",
    )
    parser.add_argument(
        "--num_explanations",
        type=int,
        default=50,
        help="The number of explanations to generate in the given dataset",
    )
    return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = load_args()
args.exp_name = f"{args.dataset}_{args.model}_{args.explanation_algorithm}"
