import os
import torch

from matplotlib import pyplot as plt

from graphxai.datasets import GraphDataset

import networkx as nx
import torch
from torch_geometric.data import Data
import random
import numpy as np

from graphxai.utils import Explanation

def create_shape_graph(shape, features='random'):
    # Number of nodes in the tree is a random number between 10 and 25
    n = random.randint(10, 25)
    
    # Generate a random tree using networkx
    tree = nx.random_tree(n=n)
    tree_nodes = list(tree.nodes)
    
    # Create the specified shape
    if shape == 'cycle_4':
        cycle = nx.cycle_graph(4)
    elif shape == 'cycle_5':
        cycle = nx.cycle_graph(5)
    elif shape == 'cycle_6':
        cycle = nx.cycle_graph(6)
    elif shape == 'cycle_8':
        cycle = nx.cycle_graph(8)
    elif shape == 'wheel':
        cycle = nx.wheel_graph(8)  # Wheel graph with 8 nodes
    elif shape == 'house':
        cycle = nx.house_graph()
    elif shape == 'cube':
        cycle = nx.cubical_graph()
    elif shape == 'peterson':
        cycle = nx.petersen_graph()
    elif shape == 'house_x':
        cycle = nx.house_x_graph()
    elif shape == 'star':
        cycle = nx.star_graph(6)
    elif shape == 'bull':
        cycle = nx.bull_graph()
    else:
        raise ValueError("Unknown shape type")
    
    # Attach the shape to a random node in the tree
    attach_node = random.choice(tree_nodes)
    
    # Shift the cycle nodes to avoid conflict with tree node indices
    cycle = nx.relabel_nodes(cycle, lambda x: x + n)
    cycle_nodes = list(cycle.nodes)
    
    # Create a new node that connects the tree and the cycle
    new_edge = (attach_node, cycle_nodes[0])
    tree.add_edges_from(cycle.edges)
    tree.add_edge(*new_edge)
    
    # Create the PyG graph
    edge_index = torch.tensor(list(tree.edges)).t().contiguous()
    # Make it undirected
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    
    # Create the edge mask
    edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    
    # Cycle edges get 1 in the mask
    for i, (u, v) in enumerate(edge_index.t().tolist()):
        if u >= n and v >= n:
            edge_mask[i] = 1
    
    # Create the node mask based on edge importance
    node_mask = torch.zeros(tree.number_of_nodes(), dtype=torch.bool)
    for u, v in edge_index.t().tolist():
        if edge_mask[edge_index.t().tolist().index([u, v])] == 1:
            node_mask[u] = 1
            node_mask[v] = 1
    
    # Generate node features
    num_nodes = tree.number_of_nodes()
    if features == 'uniform':
        node_features = torch.ones((num_nodes, 16))
    elif features == 'random':
        node_features = torch.rand((num_nodes, 16))
    else:
        raise ValueError("Unknown features type")
    
    return Data(edge_index=edge_index, x=node_features), [Explanation(edge_imp=edge_mask, node_imp=node_mask)]


def visualize_graph(data, edge_mask):
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())
    plt.figure(figsize=(24, 24))  
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=10, node_color='lightblue', font_size=10, font_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=data.edge_index.t().tolist(), edge_color='black')
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        if edge_mask[i] == 1:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color='red', width=2)
    plt.savefig('TEST.png')


def load_graphs(num_samples, shape1, shape2, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    graphs = []
    explanations = []
    
    # generate for shape1
    for i in range(num_samples//2):
        g, e = create_shape_graph(shape1)
        g.y = torch.tensor([0])
        graphs.append(g)
        explanations.append(e)
    
    # generate for shape2
    for i in range(num_samples//2):
        g, e = create_shape_graph(shape2)
        g.y = torch.tensor([1])
        graphs.append(g)
        explanations.append(e)
    
    return graphs, explanations


class Synth(GraphDataset):

    def __init__(
        self,
        num_samples,
        shape1,
        shape2,
        split_sizes=(0.8, 0.2, 0),
        seed=None,
        device=None,
    ):
        """
        Args:
            split_sizes (tuple):
            seed (int, optional):
            data_path (str, optional):
        """

        assert shape1 != shape2, "Shape1 and Shape2 must be different"

        self.graphs, self.explanations = load_graphs(
            num_samples=num_samples,
            shape1=shape1,
            shape2=shape2,
            seed=seed
        )

        # print(self.graphs)
        # print(self.explanations)

        # visualize_graph(self.graphs[0], self.explanations[0])

        super().__init__(
            name="Synth", seed=seed, split_sizes=split_sizes, device=device
        )
