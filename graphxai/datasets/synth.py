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

atom_types = ["C", "O", "F", "N", "OH", "S", "Cl", "Br"]


def create_carboxyl(num_atoms, include_carboxyl=True, num=4):
    # Atom types
    # Initialize the graph
    G = nx.Graph()
    
    # Counter for nodes added
    atom_count = 0
    
    # create a carbon chain of random length
    num_nodes_in_chain = random.randint(num_atoms//2, num_atoms - num_atoms//4)
    
    for i in range(num_nodes_in_chain):
        # Add a carbon atom
        G.add_node(atom_count, atom="C", is_carboxyl=False)
        
        # Connect the carbon atom to the previous atom
        if atom_count > 0:
            G.add_edge(atom_count, atom_count - 1)
        
        atom_count += 1
    
    
    # Add a carboxyl group if required
    if include_carboxyl:
        for i in range(num):
            if atom_count >= num_atoms:
                break
            # Create carboxyl group atoms: O, C, OH
            G.add_node(atom_count, atom="O", is_carboxyl=True)
            atom_count += 1
            G.add_node(atom_count, atom="C", is_carboxyl=True)
            atom_count += 1
            G.add_node(atom_count, atom="OH", is_carboxyl=True)
            atom_count += 1
            
            # Connect the carboxyl group atoms to each other
            G.add_edge(atom_count - 1, atom_count - 2)
            G.add_edge(atom_count - 2, atom_count - 3)
            
            # connec the carbon atom to the first atom in the chain
            G.add_edge(atom_count - 2, 0 + i)
    
    # for the remaining atoms, add them randomly and connect them to the chain; dont use C, OH or O
    
    for i in range(num_atoms - atom_count):
        atom_type = random.choice([x for x in atom_types if x not in ["C", "OH", "O"]])
        
        # Connect the atom to the chain
        node_to_connect = random.randint(0, num_nodes_in_chain - 1)
        
        # ensure the node has degree less than 4
        tries = 0
        while G.degree(node_to_connect) >= 4 and tries < 100:
            node_to_connect = random.randint(0, num_nodes_in_chain - 1)
            tries += 1
        if tries == 100:
            print("Could not add atom")
            break
        G.add_node(atom_count, atom=atom_type, is_carboxyl=False)
        G.add_edge(atom_count, node_to_connect)
        
        atom_count += 1
        
    # if num_atoms not reached, add more carbon atoms to the chain
    while atom_count < num_atoms:
        G.add_node(atom_count, atom="C", is_carboxyl=False)
        G.add_edge(atom_count, atom_count - 1)
        atom_count += 1
        
    # if we are over the number of atoms, remove the extra atoms
    
    while atom_count > num_atoms:
        atom_count -= 1
        G.remove_node(atom_count)

    # Ensure no cycles are present by converting the graph to a tree
    G = nx.minimum_spanning_tree(G)
    
    return G


def create_shape_graph(shape, features='random'):
    # Number of nodes in the tree is a random number between 10 and 25
    n = random.randint(25, 30)
    
    if shape == 'carboxyl' or shape == 'no_carboxyl':
        G = create_carboxyl(n, include_carboxyl=shape == 'carboxyl', num=1)
        # create a one hot feature vector for each node based on the atom type
        node_features = torch.zeros((n, 8))
        for i, node in G.nodes(data=True):
            index = atom_types.index(node['atom'])
            node_features[i][index] = 1
            
        # Create the PyG graph
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        
        # check if it undirected
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        
        # Create the edge mask 
        carboxyl_nodes = [i for i, node in G.nodes(data=True) if node['is_carboxyl']]
        
        edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        for i, (u, v) in enumerate(edge_index.t().tolist()):
            if u in carboxyl_nodes or v in carboxyl_nodes:
                edge_mask[i] = 1

        # Create the node mask 
        node_mask = torch.zeros(n, dtype=torch.bool)
        node_mask[carboxyl_nodes] = 1
        
        return Data(edge_index=edge_index, x=node_features), [Explanation(edge_imp=edge_mask, node_imp=node_mask)]      
    
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
    num_features = 8 if features == 'carboxyl' else 16
    
    num_nodes = tree.number_of_nodes()
    if features == 'uniform' or features == 'carboxyl':
        node_features = torch.ones((num_nodes, num_features))
    elif features == 'random' or features == 'carboxyl':
        node_features = torch.rand((num_nodes, num_features))
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
        g, e = create_shape_graph(shape2, features='carboxyl' if shape1 == 'carboxyl' else 'random')
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
