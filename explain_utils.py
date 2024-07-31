from collections import defaultdict
from pprint import pprint
from typing import List, Optional, Union
from graphxai.explainers.pgm_explainer.pgm_explainer import PGMExplainer
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import (
    Explainer,
    GNNExplainer,
    PGExplainer,
    CaptumExplainer,
    AttentionExplainer,
    GraphMaskExplainer,
    unfaithfulness,
)
from tqdm import tqdm
from torch_geometric.explain import Explanation as PyGExplanation
from data import ComplexDataset
from graphxai.utils.explanation import Explanation as GraphXAIExplanation
from graphxai.explainers import SubgraphX, PGExplainer, GNN_LRP, RandomExplainer
from graphxai.explainers._base import _BaseExplainer
from graphxai.datasets.dataset import GraphDataset, NodeDataset
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
from config import args
import os
from torch_geometric.utils import k_hop_subgraph
from config import device
import numpy as np


def get_explanation_algorithm(name):
    if name == "GNNExplainer":
        return GNNExplainer
    elif name == "PGExplainer":
        return PGExplainer
    elif name == "Captum":
        return CaptumExplainer
    elif name == "AttentionExplainer":
        return AttentionExplainer
    elif name == "GraphMaskExplainer":
        return GraphMaskExplainer
    elif name == "SubgraphX":
        return SubgraphX
    elif name == "GNN_LRP":
        return GNN_LRP
    elif name == "Random":
        return RandomExplainer
    raise NotImplementedError(f"Explanation algorithm {name} is not implemented.")


def initialise_explainer(
    model,
    explanation_algorithm_name,
    explanation_epochs=200,
    explanation_lr=0.01,
    task="binary_classification",
    node_mask_type=None,
    edge_mask_type="object",
):
    if explanation_algorithm_name == "AttentionExplainer":
        return Explainer(
            model=model,
            explanation_type=("model"),
            algorithm=get_explanation_algorithm(explanation_algorithm_name)().to(
                device
            ),
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
            model_config=dict(
                mode=task,
                task_level=args.task_level,
                return_type="probs",
            ),
        )
    elif explanation_algorithm_name == "GraphMaskExplainer":
        return Explainer(
            model=model,
            explanation_type=("model"),
            algorithm=get_explanation_algorithm(explanation_algorithm_name)(
                epochs=explanation_epochs, lr=explanation_lr, num_layers=2, log=False
            ).to(device),
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
            model_config=dict(
                mode=task,
                task_level=args.task_level,
                return_type="probs",
            ),
        )
    elif explanation_algorithm_name == "GNN_LRP":
        return GNN_LRP(model=model)
    elif explanation_algorithm_name == "Random":
        return RandomExplainer(model=model)
    elif explanation_algorithm_name != "SubgraphX":
        return Explainer(
            model=model,
            explanation_type=("model"),
            algorithm=get_explanation_algorithm(explanation_algorithm_name)(
                epochs=explanation_epochs, lr=explanation_lr, num_layers=2
            ).to(device),
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
            model_config=dict(
                mode=task,
                task_level=args.task_level,
                return_type="probs",
            ),
        )
    elif explanation_algorithm_name == "SubgraphX":
        # return SubgraphX(
        #     model=model,
        #     num_hops=2 if args.task_level == "node" else None,
        # )
        return PGMExplainer(model=model, explain_graph=True, perturb_mode="mean")


def get_graph_level_explanation(
    explainer: Union[Explainer, PGMExplainer, SubgraphX], data: Data
):
    pred = None
    if args.explanation_algorithm not in ["SubgraphX", "GNN_LRP", "Random"]:
        pred = explainer(data.x, edge_index=data.edge_index)
    elif args.explanation_algorithm in ["SubgraphX", "GNN_LRP", "Random"]:
        pred = explainer.get_explanation_graph(
            data.x, data.edge_index, forward_kwargs={"batch": data.batch}
        )
        pred = {"edge_mask": pred.edge_imp, "node_mask": pred.node_imp}
    return pred


def kl_divergence_distributions(P, Q):
    # Convert to numpy arrays for element-wise operations
    P = P.cpu().detach().numpy()
    Q = Q.cpu().detach().numpy()
    epsilon = 1e-10
    # Add epsilon to avoid log(0) and division by zero issues
    P = np.clip(P, epsilon, 1)
    Q = np.clip(Q, epsilon, 1)

    # Ensure P and Q are valid probability distributions
    if not np.all(P >= 0) or not np.all(Q >= 0):
        raise ValueError("All elements in P and Q should be non-negative")
    if not np.isclose(np.sum(P), 1) or not np.isclose(np.sum(Q), 1):
        raise ValueError("P and Q should sum to 1")

    # Compute the KL divergence
    kl_div = np.sum(P * np.log(P / Q))

    return torch.tensor(kl_div)


def explain_graph_dataset(
    explainer: Union[Explainer, _BaseExplainer],
    dataset: GraphDataset,
    num=50,
    correct_mask=None,
):
    """
    Explains the dataset using the explainer. We only explain a fraction of the dataset, as the explainer can be slow.
    """
    pred_explanations = []
    ground_truth_explanations = []
    count = 0
    f = []
    # for i in tqdm(range(num), desc="Explaining Graphs"):
    for i, index in enumerate(tqdm(dataset.test_index)):
        if count >= num:
            break

        if not correct_mask[i]:
            continue

        data, gt_explanation = dataset[index]

        if len(gt_explanation) == 0:
            continue

        zero_flag = True
        for gt in gt_explanation:
            if gt.edge_imp.sum().item() != 0:
                zero_flag = False

        if zero_flag:
            continue

        data = data.to(device)

        assert data.x is not None, "Data must have node features."
        assert data.edge_index is not None, "Data must have edge index."
        pred = get_graph_level_explanation(explainer, data)
        if args.expl_type == "edge":
            if args.explanation_aggregation == "topk":
                k = int(0.25 * len(pred["edge_mask"]))
                # take top k edges as 1 and rest as 0
                pred["edge_mask"] = (
                    pred["edge_mask"] >= pred["edge_mask"].topk(k).values.min()
                ).float()
            elif args.explanation_aggregation == "threshold":
                pred["edge_mask"] = pred["edge_mask"] >= 0.5
        elif args.expl_type == "node":
            if args.explanation_aggregation == "topk":
                k = int(0.25 * len(pred["node_mask"]))
                # take top k nodes as 1 and rest as 0
                pred["node_mask"] = (
                    pred["node_mask"] >= pred["node_mask"].topk(k).values.min()
                ).float()
            elif args.explanation_aggregation == "threshold":
                pred["node_mask"] = norm(pred["node_mask"]) >= 0.5
        else:
            raise NotImplementedError(
                f"Explanation type {args.expl_type} is not implemented."
            )
        # faithfulness = explanation_faithfulness(explainer, data, pred)
        faithfulness = torch.tensor(0.0)
        f.append(faithfulness)
        pred_explanations.append(pred)
        ground_truth_explanations.append(gt_explanation)
        count += 1
    # take mean of faithfulness and get the number out of the tensor
    f = sum(f) / len(f)
    f = f.item()
    print(f)
    return pred_explanations, ground_truth_explanations, f


def explanation_faithfulness(
    graph_explainer: Union[Explainer, _BaseExplainer],
    data: Data,
    predicted_explanation: PyGExplanation,
):
    mask = "node_mask" if args.expl_type == "node" else "edge_mask"
    predicted_explanation[mask] = predicted_explanation[mask].float()
    y = graph_explainer.get_prediction(data.x, data.edge_index)
    y_masked = graph_explainer.get_masked_prediction(
        x=data.x, edge_index=data.edge_index, edge_mask=predicted_explanation[mask]
    )
    y = torch.cat([1 - y, y])
    y_masked = torch.cat([1 - y_masked, y_masked])

    # convert y_masked to a log probability
    y_masked = F.log_softmax(y_masked, dim=0)

    kl_div = F.kl_div(y_masked, y, reduction="batchmean")
    # kl(p,q) p=log probs, q = prob

    # kl_div = kl_divergence_distributions(y, y_masked)
    return torch.exp(-kl_div)


def explanation_accuracy(
    ground_truth_explanation: List[GraphXAIExplanation],
    predicted_explanation: List[PyGExplanation],
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
        if args.expl_type == "node":
            pred_edge_mask = pred["node_mask"]
        else:
            pred_edge_mask = pred["edge_mask"]  # thresholded explanation
        best_gt_edge_mask = None
        max_gt_acc = 0
        max_gt_precision = 0
        max_gt_recall = 0
        max_gt_f1 = 0
        max_gt_jaccard = 0
        max_gt_auc = 0

        if len(gt_list) == 0:
            print("NO VALID EXPLANATION")
            continue

        # pred_gxai = GraphXAIExplanation(edge_imp=pred_edge_mask)

        # _,_,acc = graph_exp_acc_graph(gt_list, pred_gxai)
        loop_flag = False  # flag to check if the below loop has been executed
        for i, gt in enumerate(gt_list):
            try:
                if args.expl_type == "node":
                    gt_edge_mask = gt.node_imp
                else:
                    gt_edge_mask = gt.edge_imp

                if gt_edge_mask.sum().item() == 0:
                    continue

                gt_edge_mask = gt_edge_mask.cpu().numpy()
                if isinstance(pred_edge_mask, torch.Tensor):
                    pred_edge_mask = pred_edge_mask.cpu().numpy()
                # pred_edge_mask = pred_edge_mask.cpu().numpy()

                edge_mask_accuracy = accuracy_score(gt_edge_mask, pred_edge_mask)
                edge_mask_precision = precision_score(
                    gt_edge_mask, pred_edge_mask, zero_division=0
                )
                edge_mask_recall = recall_score(
                    gt_edge_mask, pred_edge_mask, zero_division=0
                )
                edge_mask_f1 = f1_score(gt_edge_mask, pred_edge_mask, zero_division=0)
                edge_mask_jaccard = jaccard_score(
                    gt_edge_mask, pred_edge_mask, zero_division=0
                )
                edge_mask_auc = roc_auc_score(gt_edge_mask, pred_edge_mask)
                if edge_mask_jaccard >= max_gt_jaccard:
                    max_gt_acc = edge_mask_accuracy
                    max_gt_precision = edge_mask_precision
                    max_gt_recall = edge_mask_recall
                    max_gt_f1 = edge_mask_f1
                    max_gt_jaccard = edge_mask_jaccard
                    max_gt_auc = edge_mask_auc
                    best_gt_edge_mask = gt_edge_mask
                loop_flag = True  # loop has been executed at least once
            except Exception as e:
                print(e)
                continue
        if not loop_flag:
            continue
        # if max_gt_jaccard == 0:
        #     print(pred_edge_mask)
        #     print(best_gt_edge_mask)
        acc += max_gt_acc
        precision += max_gt_precision
        recall += max_gt_recall
        f1 += max_gt_f1
        jaccard += max_gt_jaccard
        auc += max_gt_auc

        valid_explanations_count += 1  # increment valid explanations count as the loop has been executed at least once

    acc = acc / valid_explanations_count
    precision = precision / valid_explanations_count
    recall = recall / valid_explanations_count
    f1 = f1 / valid_explanations_count
    jaccard = jaccard / valid_explanations_count
    auc = auc / valid_explanations_count

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "auc": auc,
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
        idx_to_take = 0
        while gt_explanation[idx_to_take].edge_imp.sum().item() == 0:
            idx_to_take += 1

        gt_edge_mask = gt_explanation[idx_to_take].edge_imp.reshape(-1, 1)

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
def create_edge_mapping(graph):
    edge_type = graph.edge_type  # list of edge types for each edge
    # create a mapping {(min(u,v), max(u,v)): [edge_idx]}
    cells_to_connections = defaultdict(list)
    for i in range(len(graph.edge_index[0])):
        u, v = graph.edge_index[0][i].item(), graph.edge_index[1][i].item()
        cells_to_connections[(min(u, v), max(u, v))].append(i)

    # convert to tuple mapping (min(u,v), max(u,v)): (edge_idx)
    for key in cells_to_connections.keys():
        cells_to_connections[key] = tuple(cells_to_connections[key])

    def query_edges(node_idx):
        # get all edges of the form (node_idx, x) and (x, node_idx) where node_idx > x
        edges = []
        for i in range(len(graph.edge_index[0])):
            u, v = graph.edge_index[0][i].item(), graph.edge_index[1][i].item()
            big_idx = max(u, v)
            if big_idx == node_idx:
                edges.append((min(u, v), max(u, v)))
        edges = list(set(edges))
        return edges

    # get node_type_1 to node_type_2 edges
    conn_1_2_to_conn_0_1 = []
    for i in range(len(edge_type)):
        if edge_type[i] == 3 or edge_type[i] == 4:
            u, v = graph.edge_index[0][i].item(), graph.edge_index[1][i].item()
            mi = min(u, v)
            ma = max(u, v)
            u, v = mi, ma
            connections_0_1 = query_edges(u)
            for conn in connections_0_1:
                conn_1_2_to_conn_0_1.append(
                    (cells_to_connections[u, v], cells_to_connections[conn])
                )

    conn_0_1_to_conn_0_0 = []
    for i in range(len(edge_type)):
        if edge_type[i] == 1 or edge_type[i] == 2:
            u, v = graph.edge_index[0][i].item(), graph.edge_index[1][i].item()
            edge_idx = max(u, v)

            connections_0_1 = query_edges(edge_idx)
            x, y = connections_0_1[0][0], connections_0_1[1][0]
            edge_0_0 = (min(x, y), max(x, y))

            for conn in connections_0_1:
                conn_0_1_to_conn_0_0.append(
                    (cells_to_connections[conn], cells_to_connections[edge_0_0])
                )

    # pprint(conn_1_2_to_conn_0_1)

    # construct conn_1_2_to_conn_0_0
    conn_1_2_to_conn_0_0 = []
    for i in range(len(conn_1_2_to_conn_0_1)):
        for j in range(len(conn_0_1_to_conn_0_0)):
            if conn_1_2_to_conn_0_1[i][1] == conn_0_1_to_conn_0_0[j][0]:
                conn_1_2_to_conn_0_0.append(
                    (conn_1_2_to_conn_0_1[i][0], conn_0_1_to_conn_0_0[j][1])
                )
    conn_1_2_to_conn_0_0 = list(set(conn_1_2_to_conn_0_0))
    # pprint(conn_1_2_to_conn_0_0)

    # [(1_2_edge_up, 1_2_edge_down), (0_1_edge_up, 0_1_edge_down)]
    # =
    # [(1_2_edge_up, 0_1_edge_up), (1_2_edge_up, 0_1_edge_down), (1_2_edge_down, 0_1_edge_up), (1_2_edge_down, 0_1_edge_down)]

    conn_1_2_to_conn_0_1_temp = []
    for conn_pair in conn_1_2_to_conn_0_1:
        edge_1_2_0 = conn_pair[0][0]
        edge_1_2_1 = conn_pair[0][1]

        edge_0_1_0 = conn_pair[1][0]
        edge_0_1_1 = conn_pair[1][1]

        conn_1_2_to_conn_0_1_temp.append((edge_1_2_0, edge_0_1_0))
        conn_1_2_to_conn_0_1_temp.append((edge_1_2_0, edge_0_1_1))
        conn_1_2_to_conn_0_1_temp.append((edge_1_2_1, edge_0_1_0))
        conn_1_2_to_conn_0_1_temp.append((edge_1_2_1, edge_0_1_1))

    conn_1_2_to_conn_0_1 = conn_1_2_to_conn_0_1_temp

    conn_0_1_to_conn_0_0_temp = []
    for conn_pair in conn_0_1_to_conn_0_0:
        edge_0_1_0 = conn_pair[0][0]
        edge_0_1_1 = conn_pair[0][1]

        edge_0_0_0 = conn_pair[1][0]
        edge_0_0_1 = conn_pair[1][1]

        conn_0_1_to_conn_0_0_temp.append((edge_0_1_0, edge_0_0_0))
        conn_0_1_to_conn_0_0_temp.append((edge_0_1_0, edge_0_0_1))
        conn_0_1_to_conn_0_0_temp.append((edge_0_1_1, edge_0_0_0))
        conn_0_1_to_conn_0_0_temp.append((edge_0_1_1, edge_0_0_1))

    conn_0_1_to_conn_0_0 = conn_0_1_to_conn_0_0_temp

    conn_1_2_to_conn_0_0_temp = []
    for conn_pair in conn_1_2_to_conn_0_0:
        edge_1_2_0 = conn_pair[0][0]
        edge_1_2_1 = conn_pair[0][1]

        edge_0_0_0 = conn_pair[1][0]
        edge_0_0_1 = conn_pair[1][1]

        conn_1_2_to_conn_0_0_temp.append((edge_1_2_0, edge_0_0_0))
        conn_1_2_to_conn_0_0_temp.append((edge_1_2_0, edge_0_0_1))
        conn_1_2_to_conn_0_0_temp.append((edge_1_2_1, edge_0_0_0))
        conn_1_2_to_conn_0_0_temp.append((edge_1_2_1, edge_0_0_1))

    conn_1_2_to_conn_0_0 = conn_1_2_to_conn_0_0_temp

    return {
        "cycle_node_to_edge_node": conn_1_2_to_conn_0_1,
        "edge_node_to_edge": conn_0_1_to_conn_0_0,
        "cycle_node_to_edge": conn_1_2_to_conn_0_0,
    }


def create_node_mapping(graph):
    node_type = graph.node_type

    # create a mapping {cycle_node: [edge_node]}, {edge_node: [og_node]}, {cycle_node: [og_node]}

    cycle_node_to_edge_node = []
    edge_node_to_og_node = []
    cycle_node_to_og_node = []
    
    cache = set()

    for i in range(len(graph.edge_index[0])):
        u, v = graph.edge_index[0][i].item(), graph.edge_index[1][i].item()
        mi, ma = min(u, v), max(u, v)
        # check cache to avoid duplicate mappings
        if (mi, ma) in cache:
            continue
        cache.add((mi, ma))

        # get the node types
        u_type, v_type = node_type[mi].item(), node_type[ma].item()
    

        if u_type == v_type:
            continue  # we only care about node-to-edge_node or edge_node-to-og_node mappings

        if u_type == 0 and v_type == 1:
            og_node = u
            edge_node = v
            edge_node_to_og_node.append((edge_node, og_node))
        elif u_type == 1 and v_type == 2:
            edge_node = u
            cycle_node = v
            cycle_node_to_edge_node.append((cycle_node, edge_node))

    # get common edge nodes to create cycle_node_to_og_node mapping
    for edge in cycle_node_to_edge_node:
        edge_node = edge[1]
        for edge_pair in edge_node_to_og_node:
            if edge_pair[0] == edge_node:
                cycle_node_to_og_node.append((edge[0], edge_pair[1]))

    return {
        "cycle_node_to_edge_node": cycle_node_to_edge_node,
        "edge_node_to_node": edge_node_to_og_node,
        "cycle_node_to_node": cycle_node_to_og_node,
    }


def hierarchical_prop(graph, explanation, mappings, alpha_c=1.0, alpha_e=1.0):
    if args.expl_type == "node":
        node_mask = explanation["node_mask"]
        node_type = graph.node_type
        num_og_nodes = (node_type == 0).sum().item()
        new_node_mask = torch.zeros(num_og_nodes)
        new_node_mask = new_node_mask.to(device)

        cycle_node_to_edge_node = mappings["cycle_node_to_edge_node"]
        edge_node_to_node = mappings["edge_node_to_node"]

        for node_pair in cycle_node_to_edge_node:
            cycle_node = node_pair[0]
            edge_node = node_pair[1]
            node_mask[edge_node] += (node_mask[cycle_node] - 0.5) * alpha_c

        for node_pair in edge_node_to_node:
            edge_node = node_pair[0]
            og_node = node_pair[1]
            node_mask[og_node] += (node_mask[edge_node] - 0.5) * alpha_e

        new_node_mask[:num_og_nodes] = node_mask[:num_og_nodes]

        return new_node_mask
    else:
        edge_mask = explanation["edge_mask"]
        edge_type = graph.edge_type

        num_og_edges = (edge_type == 0).sum().item()
        new_edge_mask = torch.zeros(num_og_edges)
        new_edge_mask = new_edge_mask.to(device)
        cycle_node_to_edge_node = mappings["cycle_node_to_edge_node"]
        edge_node_to_edge = mappings["edge_node_to_edge"]

        for edge_pair in cycle_node_to_edge_node:
            edge_1_2 = edge_pair[0]
            edge_0_1 = edge_pair[1]
            edge_mask[edge_0_1] += (edge_mask[edge_1_2] - 0.5) * alpha_c

        for edge_pair in edge_node_to_edge:
            edge_0_1 = edge_pair[0]
            edge_0_0 = edge_pair[1]
            edge_mask[edge_0_0] += (edge_mask[edge_0_1] - 0.5) * alpha_e

        new_edge_mask[:num_og_edges] = edge_mask[:num_og_edges]

        return new_edge_mask


def direct_prop(graph, explanation, mappings, alpha_c=1.0, alpha_e=1.0):
    if args.expl_type == "node":
        node_mask = explanation["node_mask"]
        node_type = graph.node_type
        num_og_nodes = (node_type == 0).sum().item()
        new_node_mask = torch.zeros(num_og_nodes)
        new_node_mask = new_node_mask.to(device)

        cycle_node_to_node = mappings["cycle_node_to_node"]
        edge_node_to_node = mappings["edge_node_to_node"]

        for node_pair in cycle_node_to_node:
            cycle_node = node_pair[0]
            og_node = node_pair[1]
            node_mask[og_node] += (node_mask[cycle_node] - 0.5) * alpha_c

        for node_pair in edge_node_to_node:
            edge_node = node_pair[0]
            og_node = node_pair[1]
            node_mask[og_node] += (node_mask[edge_node] - 0.5) * alpha_e

        new_node_mask[:num_og_nodes] = node_mask[:num_og_nodes]

        return new_node_mask

    else:
        edge_mask = explanation["edge_mask"]
        edge_type = graph.edge_type

        num_og_edges = (edge_type == 0).sum().item()
        new_edge_mask = torch.zeros(num_og_edges)
        new_edge_mask = new_edge_mask.to(device)

        cycle_node_to_edge = mappings["cycle_node_to_edge"]
        edge_node_to_edge = mappings["edge_node_to_edge"]

        for edge_pair in cycle_node_to_edge:
            edge_1_2 = edge_pair[0]
            edge_0_0 = edge_pair[1]
            edge_mask[edge_0_0] += (edge_mask[edge_1_2] - 0.5) * alpha_c

        for edge_pair in edge_node_to_edge:
            edge_0_1 = edge_pair[0]
            edge_0_0 = edge_pair[1]
            edge_mask[edge_0_0] += (edge_mask[edge_0_1] - 0.5) * alpha_e

        new_edge_mask[:num_og_edges] = edge_mask[:num_og_edges]

        return new_edge_mask


def probabilistic_propagation(graph, explanation, mappings, alpha_c=1.0, alpha_e=1.0):
    edge_mask = explanation["edge_mask"]
    edge_type = graph.edge_type
    num_og_edges = (edge_type == 0).sum().item()
    new_edge_mask = torch.zeros(num_og_edges)
    new_edge_mask = new_edge_mask.to(device)

    cycle_node_to_edge_node = mappings["cycle_node_to_edge_node"]
    edge_node_to_edge = mappings["edge_node_to_edge"]

    for edge_pair in cycle_node_to_edge_node:
        edge_1_2, edge_0_1 = edge_pair
        prob = torch.sigmoid(edge_mask[edge_1_2])
        edge_mask[edge_0_1] += (prob - 0.5) * alpha_c

    for edge_pair in edge_node_to_edge:
        edge_0_1, edge_0_0 = edge_pair
        prob = torch.sigmoid(edge_mask[edge_0_1])
        edge_mask[edge_0_0] += (prob - 0.5) * alpha_e

    new_edge_mask[:num_og_edges] = edge_mask[:num_og_edges]
    return new_edge_mask


def random_noise_propagation(
    graph, explanation, alpha_c=1.0, alpha_e=1.0, noise_level=0.05
):
    edge_mask = explanation["edge_mask"]
    edge_type = graph.edge_type
    num_og_edges = (edge_type == 0).sum().item()
    new_edge_mask = torch.zeros(num_og_edges)
    new_edge_mask = new_edge_mask.to(device)

    mappings = create_edge_mapping(graph)
    cycle_node_to_edge_node = mappings["cycle_node_to_edge_node"]
    edge_node_to_edge = mappings["edge_node_to_edge"]

    for edge_pair in cycle_node_to_edge_node:
        edge_1_2, edge_0_1 = edge_pair
        noise = torch.randn(1).item() * noise_level
        edge_mask[edge_0_1] += ((edge_mask[edge_1_2] - 0.5) * alpha_c) + noise

    for edge_pair in edge_node_to_edge:
        edge_0_1, edge_0_0 = edge_pair
        noise = torch.randn(1).item() * noise_level
        edge_mask[edge_0_0] += ((edge_mask[edge_0_1] - 0.5) * alpha_e) + noise

    new_edge_mask[:num_og_edges] = edge_mask[:num_og_edges]
    return new_edge_mask


def nonlinear_activation_propagation(
    graph, explanation, mappings, alpha_c=1.0, alpha_e=1.0
):
    if args.expl_type == "node":
        node_mask = explanation["node_mask"]
        node_type = graph.node_type
        num_og_nodes = (node_type == 0).sum().item()
        new_node_mask = torch.zeros(num_og_nodes)
        new_node_mask = new_node_mask.to(device)

        cycle_node_to_node = mappings["cycle_node_to_node"]
        edge_node_to_node = mappings["edge_node_to_node"]

        for node_pair in cycle_node_to_node:
            cycle_node = node_pair[0]
            og_node = node_pair[1]
            node_mask[og_node] += torch.tanh((node_mask[cycle_node] - 0.5) * alpha_c)

        for node_pair in edge_node_to_node:
            edge_node = node_pair[0]
            og_node = node_pair[1]
            node_mask[og_node] += torch.tanh((node_mask[edge_node] - 0.5) * alpha_e)

        new_node_mask[:num_og_nodes] = node_mask[:num_og_nodes]

        return new_node_mask
    
    edge_mask = explanation["edge_mask"]
    edge_type = graph.edge_type
    num_og_edges = (edge_type == 0).sum().item()
    new_edge_mask = torch.zeros(num_og_edges)
    new_edge_mask = new_edge_mask.to(device)

    cycle_node_to_edge_node = mappings["cycle_node_to_edge_node"]
    edge_node_to_edge = mappings["edge_node_to_edge"]

    for edge_pair in cycle_node_to_edge_node:
        edge_1_2, edge_0_1 = edge_pair
        edge_mask[edge_0_1] += torch.tanh((edge_mask[edge_1_2] - 0.5) * alpha_c)

    for edge_pair in edge_node_to_edge:
        edge_0_1, edge_0_0 = edge_pair
        edge_mask[edge_0_0] += torch.tanh((edge_mask[edge_0_1] - 0.5) * alpha_e)

    new_edge_mask[:num_og_edges] = edge_mask[:num_og_edges]
    return new_edge_mask


def entropy_based_propagation(graph, explanation, mappings, alpha_c=1.0, alpha_e=1.0):
    if args.expl_type == "node":
        node_mask = explanation["node_mask"]
        node_type = graph.node_type
        num_og_nodes = (node_type == 0).sum().item()
        new_node_mask = torch.zeros(num_og_nodes)
        new_node_mask = new_node_mask.to(device)

        cycle_node_to_edge_node = mappings["cycle_node_to_edge_node"]
        edge_node_to_node = mappings["edge_node_to_node"]

        for node_pair in cycle_node_to_edge_node:
            cycle_node = node_pair[0]
            edge_node = node_pair[1]
            entropy = -node_mask[cycle_node] * torch.log(node_mask[cycle_node] + 1e-9)
            node_mask[edge_node] += (entropy - 0.5) * alpha_c

        for node_pair in edge_node_to_node:
            edge_node = node_pair[0]
            og_node = node_pair[1]
            entropy = -node_mask[edge_node] * torch.log(node_mask[edge_node] + 1e-9)
            node_mask[og_node] += (entropy - 0.5) * alpha_e

        new_node_mask[:num_og_nodes] = node_mask[:num_og_nodes]

        return new_node_mask
    
    edge_mask = explanation["edge_mask"]
    edge_type = graph.edge_type
    num_og_edges = (edge_type == 0).sum().item()
    new_edge_mask = torch.zeros(num_og_edges)
    new_edge_mask = new_edge_mask.to(device)

    cycle_node_to_edge_node = mappings["cycle_node_to_edge_node"]
    edge_node_to_edge = mappings["edge_node_to_edge"]

    for edge_pair in cycle_node_to_edge_node:
        edge_1_2, edge_0_1 = edge_pair
        entropy = -edge_mask[edge_1_2] * torch.log(edge_mask[edge_1_2] + 1e-9)
        edge_mask[edge_0_1] += (entropy - 0.5) * alpha_c

    for edge_pair in edge_node_to_edge:
        edge_0_1, edge_0_0 = edge_pair
        entropy = -edge_mask[edge_0_1] * torch.log(edge_mask[edge_0_1] + 1e-9)
        edge_mask[edge_0_0] += (entropy - 0.5) * alpha_e

    new_edge_mask[:num_og_edges] = edge_mask[:num_og_edges]
    return new_edge_mask


def norm(x):
    # x is a tensor
    return (x - x.min()) / (x.max() - x.min())


def explain_cell_complex_dataset(
    explainer: Union[Explainer, _BaseExplainer],
    dataset: ComplexDataset,
    num=50,
    correct_mask=None,
    graph_explainer=None,
):
    """
    Explains the dataset using the explainer. We only explain a fraction of the dataset, as the explainer can be slow.
    """
    print("EXPLAINING")
    pred_explanations = []
    ground_truth_explanations = []
    count = 0
    f = []
    # for i in tqdm(range(num), desc="Explaining Cell Complexes"):
    for i, index in enumerate(tqdm(dataset.test_index)):
        if count >= num:
            break

        if not correct_mask[i]:
            continue

        data, gt_explanation, _ = dataset[index]

        if len(gt_explanation) == 0:
            continue

        zero_flag = True
        for gt in gt_explanation:
            if gt.edge_imp.sum().item() != 0:
                zero_flag = False

        if zero_flag:
            continue

        data = data.to(device)

        assert data.x is not None, "Data must have node features."
        assert data.edge_index is not None, "Data must have edge index."
        pred = get_graph_level_explanation(explainer, data)

        edge_mask = None

        if args.prop_strategy == "hp_tuning":
            info_prop_methods = []
            mappings = create_edge_mapping(data)
            for prop_method in tqdm(
                [
                    "direct_prop",
                    "hierarchical_prop",
                    "nonlinear_activation_propagation",
                    "entropy_based_propagation",
                ]
            ):
                for a_c in [0, 0.5, 1.0, 1.5]:
                    for a_e in [0, 0.5, 1.0, 1.5]:
                        edge_mask = norm(
                            globals()[prop_method](
                                data, pred, mappings, alpha_c=a_c, alpha_e=a_e
                            )
                        )
                        edge_mask = (edge_mask >= 0.5).float()
                        info_prop_methods.append(
                            {
                                "prop_method": prop_method,
                                "alpha_c": a_c,
                                "alpha_e": a_e,
                                "edge_mask": edge_mask,
                            }
                        )
            pred_explanations.append(info_prop_methods)
            ground_truth_explanations.append(gt_explanation)
        else:
            mappings = create_edge_mapping(data) if args.expl_type == "edge" else create_node_mapping(data)
            if args.prop_strategy == "direct_prop":
                edge_mask = norm(
                    direct_prop(
                        data, pred, mappings, alpha_c=args.alpha_c, alpha_e=args.alpha_e
                    )
                )
            elif args.prop_strategy == "hierarchical_prop":
                edge_mask = norm(
                    hierarchical_prop(
                        data, pred, mappings, alpha_c=args.alpha_c, alpha_e=args.alpha_e
                    )
                )
            else:
                raise NotImplementedError(
                    f"Propagation strategy {args.prop_strategy} is not implemented."
                )
                
            mask = "node_mask" if args.expl_type == "node" else "edge_mask"
            if args.explanation_aggregation == "topk":
                k = int(0.25 * len(edge_mask))
                # take top k edges as 1 and rest as 0
                pred[mask] = (
                    edge_mask >= edge_mask.topk(k).values.min()
                ).float()
            elif args.explanation_aggregation == "threshold":
                pred[mask] = (edge_mask >= 0.5).float()
            if graph_explainer is not None:
                faithfulness = explanation_faithfulness(
                    graph_explainer,
                    dataset.get_underlying_graph(index).to(device),
                    pred,
                )
                f.append(faithfulness)     
        
            pred_explanations.append(pred)
            ground_truth_explanations.append(gt_explanation)
        count += 1

    if f == []:
        f = 0
    else:
        f = sum(f) / len(f)
        f = f.item()
    return pred_explanations, ground_truth_explanations, f


def explain_dataset(
    explainer: Explainer,
    dataset: Union[GraphDataset, ComplexDataset],
    num=50,
    correct_mask=None,
    graph_explainer=None,
):
    if isinstance(dataset, ComplexDataset):
        return explain_cell_complex_dataset(
            explainer, dataset, num, correct_mask, graph_explainer=graph_explainer
        )
    elif isinstance(dataset, GraphDataset):
        return explain_graph_dataset(explainer, dataset, num, correct_mask)


def explain_nodes_graphs(explainer: Explainer, data: Data, dataset: NodeDataset):
    pred_explanations = []
    gt_explanations = []
    edge_indices = []
    for i in tqdm(range(len(data.test_mask)), desc="Explaining Nodes with graphs"):
        if data.test_mask[i] == 0:
            continue
        expl = explainer(data.x, data.edge_index, index=i)
        _, edge_index, _, hard_edge_mask = k_hop_subgraph(
            i, num_hops=2, edge_index=data.edge_index
        )
        pred_edge_mask = []
        for j in range(len(hard_edge_mask)):
            if hard_edge_mask[j]:
                pred_edge_mask.append(expl["edge_mask"][j])
        pred_edge_mask = torch.stack(pred_edge_mask)

        if args.explanation_aggregation == "topk":
            k = int(0.25 * len(pred_edge_mask))
            # take top k edges as 1 and rest as 0
            pred_edge_mask = (
                pred_edge_mask >= pred_edge_mask.topk(k).values.min()
            ).float()
        elif args.explanation_aggregation == "threshold":
            pred_edge_mask = (pred_edge_mask > 0.5).float()
        expl["edge_mask"] = pred_edge_mask
        pred_explanations.append(expl)
        gt_explanations.append(dataset.get_explanation(i))
        edge_indices.append(edge_index)
    return pred_explanations, gt_explanations, edge_indices


def remove_extra_edges(num_nodes, edge_index):
    # convert to edge list
    edge_list = edge_index.t().tolist()
    # remove edges that have nodes greater than num_nodes
    edge_list = [
        edge for edge in edge_list if edge[0] < num_nodes and edge[1] < num_nodes
    ]
    # convert back to edge index
    edge_index = torch.tensor(edge_list).t().contiguous()
    return edge_index


def explain_nodes_complex(
    explainer: Explainer, data: Data, dataset: NodeDataset, mapping
):
    pred_explanations = []
    gt_explanations = []
    edge_indices = []
    type_0_nodes = (data.node_type == 0).sum().item()
    for i in tqdm(range(len(data.test_mask)), desc="Explaining Nodes with complexes"):
        if data.test_mask[i] == 0:
            continue

        expl = explainer(data.x, data.edge_index, index=i)
        _, edge_index, _, hard_edge_mask = k_hop_subgraph(
            i,
            num_hops=2,
            edge_index=data.edge_index,
        )
        edge_index = remove_extra_edges(type_0_nodes, edge_index)
        std_edge_mask = (direct_prop(data, expl, mapping) / 3.5).tanh()

        pred_edge_mask = []
        for j in range(len(std_edge_mask)):
            if hard_edge_mask[j]:
                pred_edge_mask.append(expl["edge_mask"][j])
        pred_edge_mask = torch.stack(pred_edge_mask)
        if args.explanation_aggregation == "topk":
            k = int(0.25 * len(pred_edge_mask))
            # take top k edges as 1 and rest as 0
            pred_edge_mask = (
                pred_edge_mask >= pred_edge_mask.topk(k).values.min()
            ).float()
        elif args.explanation_aggregation == "threshold":
            pred_edge_mask = (pred_edge_mask > 0.5).float()
        expl["edge_mask"] = pred_edge_mask
        pred_explanations.append(expl)
        gt_explanations.append(dataset.get_explanation(i))
        edge_indices.append(edge_index)
    return pred_explanations, gt_explanations, edge_indices


def explain_nodes(explainer: Explainer, data: Data, dataset, mapping=None, type="g"):
    if type == "g":
        return explain_nodes_graphs(explainer, data, dataset)
    elif type == "c":
        return explain_nodes_complex(explainer, data, dataset, mapping)
    else:
        raise NotImplementedError(
            f"Node explanation for type {type} is not implemented."
        )


def save_to_graphml(data, explanation, outdir, fname, is_gt=False):
    edge_list = data.edge_index.t().tolist()
    edge_mask = None
    if is_gt:
        print(len(explanation))
        for i in range(len(explanation)):
            if explanation[i].edge_imp.sum().item() == 0:
                continue
            else:
                edge_mask = explanation[i].edge_imp.cpu().numpy().tolist()
                break
        if edge_mask is None:
            edge_mask = explanation[0].edge_imp.cpu().numpy().tolist()
    else:
        edge_mask = explanation["edge_mask"].tolist()
    G = nx.Graph()
    weighted_edges = [
        (edge_list[i][0], edge_list[i][1], edge_mask[i]) for i in range(len(edge_list))
    ]
    G.add_weighted_edges_from(weighted_edges)
    out_path = os.path.join(outdir, f"{args.current_seed}", fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nx.write_graphml(G, out_path)


def save_node_graph_to_graphml(
    edge_index, explanation, node_idx, outdir, fname, is_gt=False
):
    edge_list = edge_index.t().tolist()
    edge_mask = None

    # count number of nodes

    if is_gt:
        enc_subgraph = explanation[0].enc_subgraph
        edge_index = enc_subgraph.edge_index
        edge_list = edge_index.t().tolist()
        edge_mask = enc_subgraph.edge_mask.cpu().numpy().tolist()
    else:
        edge_mask = explanation["edge_mask"].tolist()
    G = nx.Graph()
    weighted_edges = [
        (edge_list[i][0], edge_list[i][1], edge_mask[i]) for i in range(len(edge_list))
    ]
    G.add_weighted_edges_from(weighted_edges)
    node_attr = {}
    for i in G.nodes:
        if i == node_idx:
            node_attr[i] = 1
        else:
            node_attr[i] = 0
    nx.set_node_attributes(G, {i: {"node_attr": node_attr[i]} for i in G.nodes})
    out_path = os.path.join(outdir, f"{args.current_seed}", fname)
    # create directory if it does not exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nx.write_graphml(G, out_path)
