from typing import Tuple, Union
from graphxai.datasets import Benzene, AlkaneCarbonyl, GraphDataset, Mutagenicity, FluorideCarbonyl
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader
import torch
import networkx as nx
import numpy as np


def load_dataset(name="Benzene"):
    if name == "Benzene":
        return Benzene()
    elif name == "AlkaneCarbonyl":
        return AlkaneCarbonyl()
    elif name == "Mutagenicity":
        return Mutagenicity(root="data/")
    elif name == "FluorideCarbonyl":
        return FluorideCarbonyl()
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")


def graph_to_complex(g):
    hg = HeteroData()

    if hasattr(g, "y"):
        hg.y = g.y

    # transfer nodes and original edges
    hg["node"].x = g.x
    hg["node", "to", "node"].edge_index = g.edge_index

    # adding edge nodes, INFO: new edge-node for both directions
    if g.edge_attr is not None:
        hg["edge_node"].x = g.edge_attr
    else:
        edge_x = []
        for i in range(g.edge_index.shape[1]):
            # get the constituent nodes of the edge
            source_node = g.edge_index[0][i]
            target_node = g.edge_index[1][i]
            # add the mean
            edge_x.append((g.x[source_node] + g.x[target_node]) / 2)
        hg["edge_node"].x = torch.stack(edge_x, dim=0)

    num_edges = g.edge_index.shape[1]
    range_edges = torch.arange(num_edges)  # the id of the new edges
    source_nodes = g.edge_index[0]
    target_nodes = g.edge_index[1]

    # new edge index to original node to node edge index
    mapping = {1: {}, 2: {}, 3: {}, 4: {}}

    hg["node", "to", "edge_node"].edge_index = torch.stack(
        [source_nodes, range_edges], dim=0
    )
    hg["edge_node", "to", "node"].edge_index = torch.stack(
        [range_edges, target_nodes], dim=0
    )

    for i in range(num_edges):
        mapping[1][i] = i

    for i in range(num_edges):
        mapping[2][i] = i

    # convert to networkx graph to find cycles
    nx_graph = nx.Graph()
    edge_list = g.edge_index.t().tolist()
    nx_graph.add_edges_from(edge_list)
    cycles = nx.cycle_basis(nx_graph)
    cycles = [c for c in cycles if len(c) <= 8]

    # process cycles to add to HeteroData
    cycle_nodes = []
    cycle_node_to_node = []
    node_to_cycle_node = []
    cycle_node_to_edge_node = []
    edge_node_to_cycle_node = []

    if len(cycles) == 0:
        return hg, mapping

    c_t_e_counter = 0
    e_t_c_counter = 0

    for i, cycle in enumerate(cycles):
        # create cycle node
        cycle_length = len(cycle)
        cycle_feature = np.zeros((8,))
        cycle_feature[cycle_length - 1] = 1  # one-hot
        cycle_nodes.append(cycle_feature)

        edges_in_cycle = []
        c_t_e_edges = []
        e_t_c_edges = []

        # connect cycle_node to all nodes in the cycle
        for node in cycle:
            cycle_node_to_node.append([i, node])
            node_to_cycle_node.append([node, i])

        # connect cycle_node to all edges in the cycle
        for j in range(cycle_length):
            edge = (cycle[j], cycle[(j + 1) % cycle_length])
            edge = (min(edge), max(edge))  # as edges are undirected in nx
            edge_index = edge_list.index(list(edge))
            cycle_node_to_edge_node.append([i, edge_index])
            edge_node_to_cycle_node.append([edge_index, i])

            edges_in_cycle.append(edge_index)
            c_t_e_edges.append(c_t_e_counter)
            e_t_c_edges.append(e_t_c_counter)

            # mapping[3][c_t_e_counter] = edge_index
            # mapping[4][e_t_c_counter] = edge_index

            c_t_e_counter += 1
            e_t_c_counter += 1
        
        for edge in c_t_e_edges:
            mapping[3][edge] = edges_in_cycle
        
        for edge in e_t_c_edges:
            mapping[4][edge] = edges_in_cycle

    hg["cycle_node"].x = torch.tensor(np.array(cycle_nodes), dtype=torch.float32)
    # hg['cycle_node', 'to', 'node'].edge_index = torch.tensor(np.array(cycle_node_to_node), dtype=torch.long).t()
    # hg['node', 'to', 'cycle_node'].edge_index = torch.tensor(np.array(node_to_cycle_node), dtype=torch.long).t()
    hg["cycle_node", "to", "edge_node"].edge_index = torch.tensor(
        np.array(cycle_node_to_edge_node), dtype=torch.long
    ).t()
    hg["edge_node", "to", "cycle_node"].edge_index = torch.tensor(
        np.array(edge_node_to_cycle_node), dtype=torch.long
    ).t()

    return hg, mapping


def in_house_to_homogeneous(hg):
    homo_g = hg.to_homogeneous()
    
    x = hg['node']['x']
    y = hg['edge_node']['x']
    if hg['cycle_node']:
        z = hg['cycle_node']['x']
    else:
        z = torch.zeros((0, 8))

    q = x.size(1) + y.size(1) + z.size(1)

    x_padded = torch.cat([x, torch.zeros(x.size(0), q - x.size(1))], dim=1)
    y_padded = torch.cat([torch.zeros(y.size(0), x.size(1)), y, torch.zeros(y.size(0), z.size(1))], dim=1)
    z_padded = torch.cat([torch.zeros(z.size(0), q - z.size(1)), z], dim=1)

    # concat across dim 0
    features = torch.cat([x_padded, y_padded, z_padded], dim=0)
    assert homo_g.x.size(0) == features.size(0)

    homo_g.x = features
    return homo_g


class ComplexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hg, mapping = graph_to_complex(self.dataset[idx][0])
        # print(hg)
        gt_explanation = self.dataset[idx][1]
        hg = hg.to_homogeneous()
        # hg = in_house_to_homogeneous(hg)
        # print('HOMO', hg)
        return (hg, gt_explanation, mapping)


def lift_dataset(dataset: GraphDataset) -> ComplexDataset:
    """
    Lifts the graphs in the dataset to higher order cell complexes.

    Args:
        dataset (GraphDataset): The graph dataset to lift.

    Returns:
        ComplexDataset: The cell complex dataset.
    """
    return ComplexDataset(dataset)


def load_dataset_as_complex(name="Benzene") -> ComplexDataset:
    """
    Loads the dataset and converts it to a cell complex dataset.
    """
    return lift_dataset(load_dataset(name))


def get_graph_data_loaders(dataset: GraphDataset, batch_size=32) -> Tuple[DataLoader]:
    """
    Returns the data loaders for graph datasets.
    """
    return dataset.get_train_loader(batch_size)[0], dataset.get_test_loader()[0]


def get_complex_data_loaders(
    dataset, batch_size=32, train_frac=0.8
) -> Tuple[DataLoader]:
    """
    Returns the data loaders for cell complex datasets.
    """
    # data loader
    train_data = [dataset[i][0] for i in range(int(len(dataset) * train_frac))]
    test_data = [
        dataset[i][0] for i in range(int(len(dataset) * train_frac), len(dataset))
    ]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_data_loaders(
    dataset: Union[GraphDataset, ComplexDataset], batch_size=32
) -> Tuple[DataLoader]:
    """
    Returns the data loaders for the dataset.
    """
    if isinstance(dataset, GraphDataset):
        return get_graph_data_loaders(dataset, batch_size)
    elif isinstance(dataset, ComplexDataset):
        return get_complex_data_loaders(dataset, batch_size)
