from typing import List, Optional, Union
import torch
from torch_geometric.explain import (
    Explainer,
    GNNExplainer,
    PGExplainer,
    CaptumExplainer,
)
from tqdm import tqdm
from torch_geometric.explain import Explanation as PyGExplanation
from data import ComplexDataset
from graphxai.utils.explanation import Explanation as GraphXAIExplanation
from graphxai.datasets.dataset import GraphDataset
from graphxai.metrics.metrics_graph import graph_exp_acc_graph
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import networkx as nx
from config import device
import os


def get_explanation_algorithm(name):
    if name == "GNNExplainer":
        return GNNExplainer
    elif name == "PGExplainer":
        return PGExplainer
    elif name == "Captum":
        return CaptumExplainer
    raise NotImplementedError(f"Explanation algorithm {name} is not implemented.")


def initialise_explainer(
    model,
    explanation_algorithm_name,
    explanation_epochs=200,
    explanation_lr=0.01,
    task="binary_classification",
    node_mask_type="object",
    edge_mask_type="object",
):
    if explanation_algorithm_name == "PGExplainer":
        node_mask_type = None
    return Explainer(
        model=model,
        explanation_type=(
            "model" if explanation_algorithm_name == "GNNExplainer" else "phenomenon"
        ),
        algorithm=get_explanation_algorithm(explanation_algorithm_name)(
            epochs=explanation_epochs,
            lr=explanation_lr,
        ),
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=dict(
            mode=task,
            task_level="graph",
            return_type="probs",
        ),
    )


def explain_graph_dataset(explainer: Explainer, dataset: GraphDataset, num=50):
    """
    Explains the dataset using the explainer. We only explain a fraction of the dataset, as the explainer can be slow.
    """
    pred_explanations = []
    ground_truth_explanations = []
    for i in tqdm(range(num)):
        data, gt_explanation = dataset[i]

        assert data.x is not None, "Data must have node features."
        assert data.edge_index is not None, "Data must have edge index."
        pred = explainer(
            data.x, edge_index=data.edge_index, batch=data.batch
        )
        pred["node_mask"] = pred["node_mask"] > 0.5
        pred["edge_mask"] = pred["edge_mask"] > 0.5
        pred_explanations.append(pred)
        ground_truth_explanations.append(gt_explanation)
    return pred_explanations, ground_truth_explanations


def explanation_accuracy(
    ground_truth_explanation: List[GraphXAIExplanation],
    predicted_explanation: PyGExplanation,
):
    """
    Computes the accuracy of the predicted explanation. Only works with thresholded explanations for now.
    """
    acc = 0
    precision = 0
    recall = 0
    f1 = 0
    jaccard = 0
    auc = 0
    valid_explanations_count = 0

    for pred, gt_list in zip(predicted_explanation, ground_truth_explanation):
        pred_edge_mask = pred["edge_mask"]  # thresholded explanation
        # best_gt_edge_mask = None
        # max_gt_acc = 0
        # max_gt_precision = 0
        # max_gt_recall = 0
        # max_gt_f1 = 0
        # max_gt_jaccard = 0
        # max_gt_auc = 0
        
        pred_explanation = GraphXAIExplanation(
            edge_imp=pred_edge_mask
        )
        
        acc += graph_exp_acc_graph(
            gt_exp=gt_list,
            generated_exp=pred_explanation)[2]

        # if len(gt_list) == 0:
        #     continue

        # loop_flag = False  # flag to check if the below loop has been executed
        # for i, gt in enumerate(gt_list):
        #     try:
        #         gt_edge_mask = gt.edge_imp

        #         edge_mask_accuracy = accuracy_score(gt_edge_mask, pred_edge_mask)
        #         edge_mask_precision = precision_score(
        #             gt_edge_mask, pred_edge_mask, zero_division=0
        #         )
        #         edge_mask_recall = recall_score(
        #             gt_edge_mask, pred_edge_mask, zero_division=0
        #         )
        #         edge_mask_f1 = f1_score(gt_edge_mask, pred_edge_mask, zero_division=0)
        #         edge_mask_jaccard = jaccard_score(
        #             gt_edge_mask, pred_edge_mask, zero_division=0
        #         )
        #         edge_mask_auc = roc_auc_score(gt_edge_mask, pred_edge_mask)
        #         if edge_mask_jaccard >= max_gt_jaccard:
        #             max_gt_acc = edge_mask_accuracy
        #             max_gt_precision = edge_mask_precision
        #             max_gt_recall = edge_mask_recall
        #             max_gt_f1 = edge_mask_f1
        #             max_gt_jaccard = edge_mask_jaccard
        #             max_gt_auc = edge_mask_auc
        #             best_gt_edge_mask = gt_edge_mask
        #         loop_flag = True  # loop has been executed at least once
        #     except:
        #         continue
        # if not loop_flag:
        #     continue
        # if max_gt_jaccard == 0:
        #     print(pred_edge_mask)
        #     print(best_gt_edge_mask)
        # acc += max_gt_acc
        # precision += max_gt_precision
        # recall += max_gt_recall
        # f1 += max_gt_f1
        # jaccard += max_gt_jaccard
        # auc += max_gt_auc

        # valid_explanations_count += 1  # increment valid explanations count as the loop has been executed at least once

    acc = acc / len(predicted_explanation)
    # precision = precision / valid_explanations_count
    # recall = recall / valid_explanations_count
    # f1 = f1 / valid_explanations_count
    # jaccard = jaccard / valid_explanations_count
    # auc = auc / valid_explanations_count

    return {
        "accuracy": acc,
        # "precision": precision,
        # "recall": recall,
        # "f1": f1,
        # "jaccard": jaccard,
        # "auc": auc,
    }


def visualise_explanation(
    pred_explanation: PyGExplanation,
    gt_explanation: Optional[GraphXAIExplanation] = None,
    type="edge",
    save_img=True,
):
    """
    Visualises the explanation on the graph.
    """
    assert type == "edge", "Only edge explanations are supported for now."

    if gt_explanation is not None:
        # plot both predicted and ground truth explanations side by side
        pred_edge_mask = pred_explanation["edge_mask"].reshape(-1, 1)
        gt_edge_mask = gt_explanation.edge_imp.reshape(-1, 1)

        # heatmap for predicted explanation
        plt.figure(figsize=(6, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(pred_edge_mask, cmap="hot", interpolation="nearest")
        plt.xticks([])
        plt.colorbar()
        plt.title("Predicted Explanation")

        # heatmap for ground truth explanation
        plt.subplot(1, 2, 2)
        plt.imshow(gt_edge_mask, cmap="hot", interpolation="nearest")
        plt.title("Ground Truth Explanation")
        # remove y axis ticks
        plt.xticks([])
        plt.colorbar()
        if save_img:
            plt.savefig(f"figs/fig_{time()}.png", format="png")
        # plt.show()
    else:
        # plot only predicted explanation
        pred_edge_mask = pred_explanation["edge_mask"].reshape(-1, 1)
        plt.figure(figsize=(5, 5))
        sns.heatmap(pred_edge_mask, cmap="coolwarm")
        plt.title("Predicted Explanation")
        plt.colorbar()
        if save_img:
            plt.savefig(f"figs/fig_{time()}.png", format="png")
        # plt.show()


#### FOR CELL COMPLEXES ####


def to_standard(graph, explanation, mapping):
    edge_mask = explanation["edge_mask"]
    edge_type = graph.edge_type

    num_og_edges = (edge_type == 0).sum().item()
    new_edge_mask = torch.zeros(num_og_edges)

    # print(num_og_edges, new_edge_mask)

    last_seen = -1
    counter = 0
    for i in range(len(edge_type)):
        if edge_type[i] == 0:
            new_edge_mask[i] = edge_mask[i]

        if edge_type[i] == 1:
            if last_seen != 1:
                counter = 0
            else:
                counter += 1

            idx = mapping[1][counter]
            new_edge_mask[idx] += edge_mask[i]
        elif edge_type[i] == 2:
            if last_seen != 2:
                counter = 0
            else:
                counter += 1

            idx = mapping[2][counter]
            new_edge_mask[idx] += edge_mask[i]
        elif edge_type[i] == 3:
            if last_seen != 3:
                counter = 0
            else:
                counter += 1

            # print('ERRROR', i)
            # print(edge_type[i], edge_type[i + 1])
            idx = mapping[3][counter]
            new_edge_mask[idx] += edge_mask[i]
        elif edge_type[i] == 4:
            if last_seen != 4:
                counter = 0
            else:
                counter += 1

            idx = mapping[4][counter]
            new_edge_mask[idx] += edge_mask[i]

        last_seen = edge_type[i]

    return new_edge_mask


def explain_cell_complex_dataset(explainer: Explainer, dataset: ComplexDataset, num=50):
    """
    Explains the dataset using the explainer. We only explain a fraction of the dataset, as the explainer can be slow.
    """
    pred_explanations = []
    ground_truth_explanations = []
    for i in tqdm(range(num)):
        data, gt_explanation, mapping = dataset[i]
        assert data.x is not None, "Data must have node features."
        assert data.edge_index is not None, "Data must have edge index."
        pred = explainer(
            data.x, edge_index=data.edge_index, batch=data.batch
        )
        edge_mask = (to_standard(data, pred, mapping) / 3.5).tanh() > 0.5
        pred["edge_mask"] = edge_mask
        pred_explanations.append(pred)
        ground_truth_explanations.append(gt_explanation)
    return pred_explanations, ground_truth_explanations


def explain_dataset(
    explainer: Explainer, dataset: Union[GraphDataset, ComplexDataset], num=50
):
    if isinstance(dataset, ComplexDataset):
        return explain_cell_complex_dataset(explainer, dataset, num)
    elif isinstance(dataset, GraphDataset):
        return explain_graph_dataset(explainer, dataset, num)


def save_to_graphml(data, explanation, outdir, fname, is_gt=False):
    edge_list = data.edge_index.t().tolist()
    if is_gt:
        edge_mask = explanation.edge_imp.cpu().numpy().tolist()
        print(edge_mask)
    else:
        edge_mask = explanation["edge_mask"].tolist()
    G = nx.Graph()
    weighted_edges = [
        (edge_list[i][0], edge_list[i][1], edge_mask[i]) for i in range(len(edge_list))
    ]
    G.add_weighted_edges_from(weighted_edges)
    out_path = os.path.join(outdir, fname)
    nx.write_graphml(G, out_path)
