import torch
import argparse
from time import gmtime, strftime, time


def load_args():
    parser = argparse.ArgumentParser(
        description="HOGE: Higher Order Graph Explanations"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Benzene",
        help="The dataset to use",
        choices=[
            "Benzene",
            "AlkaneCarbonyl",
            "Mutagenicity",
            "FluorideCarbonyl",
            "Synth",
            "Synth_house_wheel",
            "Synth_wheel_cube",
            "Synth_cycle_6_house",
            "Synth_cycle_4_star",
            "Synth_bull_4_cycle",
            "Synth_carboxyl_6_cycle",
            "Synth_carboxyl",
        ],
    )
    parser.add_argument(
        '--synth_shape_1',
        type=str,
        default='cycle_6',
        help='The shape of the first class in the synthetic dataset',
        choices=['cycle_4', 'cycle_5', 'cycle_6', 'cycle_8', 'wheel', 'house', 'cube', 'peterson', 'house_x', 'star', 'bull', 'carboxyl']
    )
    parser.add_argument(
        '--synth_shape_2',
        type=str,
        default='star',
        help='The shape of the second class in the synthetic dataset',
        choices=['cycle_4', 'cycle_5', 'cycle_6', 'cycle_8', 'wheel', 'house', 'cube', 'peterson', 'house_x', 'star', 'bull', "no_carboxyl"]
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        help="The model to use",
        choices=["GCN", "GAT", "GIN"],
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
        '--expl_type',
        type=str,
        default='edge',
        help='The type of explanation mask to generate',
        choices=['edge', 'node']
    )
    parser.add_argument(
        "--explanation_algorithm",
        type=str,
        default="GNNExplainer",
        help="The explanation algorithm to use",
        choices=[
            "GNNExplainer",
            "GraphMaskExplainer",
            "PGMExplainer",
            "Random",
            "GradExplainer",
            "GuidedBP",
            "SubgraphX",
        ],
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The batch size for training"
    )
    parser.add_argument(
        "--graph_epochs",
        type=int,
        default=30,
        help="The number of epochs for training on graphs",
    )
    parser.add_argument(
        "--complex_epochs",
        type=int,
        default=30,
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
        default="explanations",
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
        default=100,
        help="The number of explanations to generate in the given dataset",
    )
    parser.add_argument(
        "--task_level",
        type=str,
        default="graph",
        help="The level of the explanation task",
        choices=["graph", "node"],
    )
    parser.add_argument(
        "--explanation_aggregation",
        type=str,
        default="threshold",
        help="The aggregation method for node-level explanations",
        choices=["threshold", "topk"],
    )
    parser.add_argument(
        "--test_graph_train_complex_dataset",
        type=bool,
        default=False,
        help="Whether to test explainer with model trained on graph, and providing complex dataset",
    )
    parser.add_argument(
        "--test_complex_train_graph_dataset",
        type=bool,
        default=False,
        help="Whether to test explainer with model trained on complexes, and providing graph dataset",
    )
    parser.add_argument(
        "--remove_type_2_nodes",
        type=bool,
        default=False,
        help="Whether to remove type 2 nodes (cycles) from the dataset",
    )
    parser.add_argument(
        "--remove_type_1_nodes",
        type=bool,
        default=False,
        help="Whether to remove type 1 nodes (edges) from the dataset",
    )
    parser.add_argument(
        "--prop_strategy",
        type=str,
        default="direct_prop",
        help="The strategy for propagating the higher order explanations",
        choices=["direct_prop", "hierarchical_prop", "hp_tuning"],
    )
    parser.add_argument(
        "--alpha_c",
        type=float,
        default=0.5,
        help="The weight for the 2-complexes",
    )
    parser.add_argument(
        "--alpha_e",
        type=float,
        default=0.5,
        help="The weight for the 1-complexes",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="The random seed to start with",
    )
    parser.add_argument(
        "--end_seed",
        type=int,
        default=1,
        help="The random seed to end with (+ 1). Use end_seed = start_seed + 1 for a single seed",
    )

    return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
args = load_args()
args.exp_name = (
    f"{args.task_level}_{args.dataset}_{args.model}_{args.explanation_algorithm}"
)
args.time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

if args.explanation_algorithm in ["GradExplainer", "GuidedBP", "PGMExplainer"]:
    args.expl_type = "node"
