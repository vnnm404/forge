from collections import defaultdict
from typing import Tuple, Union
from graphxai.datasets import (
    Benzene,
    AlkaneCarbonyl,
    GraphDataset,
    Mutagenicity,
    FluorideCarbonyl,
    Synth,
)
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader
import torch
import networkx as nx
import numpy as np
import networkx as nx
from config import args
from torch_geometric.utils import to_networkx
# from config import args


def load_dataset(name="Benzene", seed=0):
    if name == "Benzene":
        return Benzene(seed=seed)
    elif name == "AlkaneCarbonyl":
        return AlkaneCarbonyl(seed=seed)
    elif name == "Mutagenicity":
        return Mutagenicity(root="data/", seed=seed)
    elif name == "FluorideCarbonyl":
        return FluorideCarbonyl(seed=seed)
    elif name[:5] == "Synth":
        return Synth(seed=seed, shape1=args.synth_shape_1, shape2=args.synth_shape_2, num_samples=2000)
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")


def graph_to_complex(g):
    hg = HeteroData()

    if hasattr(g, "y"):
        hg.y = g.y

    # transfer nodes and original edges
    hg["node"].x = g.x
    hg["node", "to0", "node"].edge_index = g.edge_index

    # edge -> (u, v) -> p(edge_node) (0 index) (u, p) and (p, v) and (p, u) and (v, p)
    node_u, node_v = g.edge_index
    edge_id = torch.arange(node_u.size(0))
    cache = defaultdict(list)
    mapper = defaultdict(lambda: defaultdict(list))
    # mapper = defaultdict(lambda: defaultdict(list))

    # edges
    node_to_edgenode = ([], [])
    edgenode_to_node = ([], [])
    edge_feat = []  # index -> [features]
    edgenode_id = 0
    for i in range(len(node_u)):
        u = node_u[i].item()
        v = node_v[i].item()
        id = edge_id[i].item()
        try:
            feat = g.edge_attr[id]
        except:
            # average of the node features
            feat = (g.x[u] + g.x[v]) / 2
        if (min(u, v), max(u, v)) in cache:
            mapper["edge_node_to_edge"][cache[(min(u, v), max(u, v))]].append(id)
            continue

        cache[(min(u, v), max(u, v))] = edgenode_id
        mapper["edge_node_to_edge"][edgenode_id].append(id)

        node_to_edgenode[0].append(u)
        node_to_edgenode[1].append(edgenode_id)
        edgenode_to_node[0].append(edgenode_id)
        edgenode_to_node[1].append(u)

        node_to_edgenode[0].append(v)
        node_to_edgenode[1].append(edgenode_id)
        edgenode_to_node[0].append(edgenode_id)
        edgenode_to_node[1].append(v)

        edge_feat.append(feat)
        edgenode_id += 1

    hg["edge_node"].x = torch.stack(edge_feat)
    hg["node", "to1", "edge_node"].edge_index = torch.stack(
        [torch.tensor(node_to_edgenode[0]), torch.tensor(node_to_edgenode[1])], dim=0
    )
    hg["edge_node", "to2", "node"].edge_index = torch.stack(
        [torch.tensor(edgenode_to_node[0]), torch.tensor(edgenode_to_node[1])], dim=0
    )

    # convert to networkx graph to find cycles
    nx_graph = nx.Graph()
    # print("EDGE", g.edge_index.t())
    edge_list = g.edge_index.t().tolist()
    # print(edge_list)
    nx_graph.add_edges_from(edge_list)
    cycles = nx.cycle_basis(nx_graph)
    cycles = [c for c in cycles if len(c) <= 8]

    # # process cycles to add to HeteroData
    cycle_nodes = []
    cycle_node_to_node = []
    node_to_cycle_node = []
    cycle_node_to_edge_node = []
    edge_node_to_cycle_node = []

    if len(cycles) == 0:
        return hg, mapper

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
        og_cte_to_edge = {}
        og_etc_to_edge = {}

        # connect cycle_node to all nodes in the cycle
        for node in cycle:
            cycle_node_to_node.append([i, node])
            node_to_cycle_node.append([node, i])

        # connect cycle_node to all edges in the cycle
        for j in range(cycle_length):
            edge = (cycle[j], cycle[(j + 1) % cycle_length])
            edge = (min(edge), max(edge))  # as edges are undirected in nx
            # edge_index = edge_list.index(list(edge))
            u = min(edge)
            v = max(edge)
            # u -> multiple edge nodes (edge node -> cycle)
            # ui = node_to_edgenode[0].index(u)
            # edge_index = node_to_edgenode[1][ui]
            # print(edge_index)

            uids = [
                node_to_edgenode[1][idx]
                for idx, u2 in enumerate(node_to_edgenode[0])
                if u2 == u
            ]
            vids = [
                node_to_edgenode[1][idx]
                for idx, v2 in enumerate(node_to_edgenode[0])
                if v2 == v
            ]

            # common id
            edge_index = set(uids).intersection(set(vids)).pop()

            cycle_node_to_edge_node.append([i, edge_index])
            edge_node_to_cycle_node.append([edge_index, i])

            edges_in_cycle.append(edge_index)
            c_t_e_edges.append(c_t_e_counter)
            e_t_c_edges.append(e_t_c_counter)
            # og_cte_to_edge.append(edge_index)
            # og_etc_to_edge.append(edge_index)
            og_cte_to_edge[c_t_e_counter] = edge_index
            og_etc_to_edge[e_t_c_counter] = edge_index

            # mapping[3][c_t_e_counter] = edge_index
            # mapping[4][e_t_c_counter] = edge_index
            mapper["cycle_node_to_edge_node"][i].append(edge_index)

            c_t_e_counter += 1
            e_t_c_counter += 1

    hg["cycle_node"].x = torch.tensor(np.array(cycle_nodes), dtype=torch.float32)
    hg["cycle_node", "to3", "edge_node"].edge_index = torch.tensor(
        np.array(cycle_node_to_edge_node), dtype=torch.long
    ).t()
    hg["edge_node", "to4", "cycle_node"].edge_index = torch.tensor(
        np.array(edge_node_to_cycle_node), dtype=torch.long
    ).t()

    return hg, mapper


def graph_to_full_complex(g):
    hg = HeteroData()

    if hasattr(g, "y"):
        hg.y = g.y

    # transfer nodes and original edges
    hg["node"].x = g.x
    hg["node", "to0", "node"].edge_index = g.edge_index

    # edge -> (u, v) -> p(edge_node) (0 index) (u, p) and (p, v) and (p, u) and (v, p)
    node_u, node_v = g.edge_index
    edge_id = torch.arange(node_u.size(0))
    cache = defaultdict(list)
    mapper = defaultdict(lambda: defaultdict(list))
    # mapper = defaultdict(lambda: defaultdict(list))

    # edges
    node_to_edgenode = ([], [])
    edgenode_to_node = ([], [])
    edge_feat = []  # index -> [features]
    edgenode_id = 0
    for i in range(len(node_u)):
        u = node_u[i].item()
        v = node_v[i].item()
        id = edge_id[i].item()
        try:
            feat = g.edge_attr[id]
        except:
            # average of the node features
            feat = (g.x[u] + g.x[v]) / 2
        if (min(u, v), max(u, v)) in cache:
            mapper["edge_node_to_edge"][cache[(min(u, v), max(u, v))]].append(id)
            continue

        cache[(min(u, v), max(u, v))] = edgenode_id
        mapper["edge_node_to_edge"][edgenode_id].append(id)

        node_to_edgenode[0].append(u)
        node_to_edgenode[1].append(edgenode_id)
        edgenode_to_node[0].append(edgenode_id)
        edgenode_to_node[1].append(u)

        node_to_edgenode[0].append(v)
        node_to_edgenode[1].append(edgenode_id)
        edgenode_to_node[0].append(edgenode_id)
        edgenode_to_node[1].append(v)

        edge_feat.append(feat)
        edgenode_id += 1

    hg["edge_node"].x = torch.stack(edge_feat)
    hg["node", "to1", "edge_node"].edge_index = torch.stack(
        [torch.tensor(node_to_edgenode[0]), torch.tensor(node_to_edgenode[1])], dim=0
    )
    hg["edge_node", "to2", "node"].edge_index = torch.stack(
        [torch.tensor(edgenode_to_node[0]), torch.tensor(edgenode_to_node[1])], dim=0
    )

    # convert to networkx graph to find cycles
    nx_graph = nx.Graph()
    # print("EDGE", g.edge_index.t())
    edge_list = g.edge_index.t().tolist()
    # print(edge_list)
    nx_graph.add_edges_from(edge_list)
    cycles = nx.cycle_basis(nx_graph)
    cycles = [c for c in cycles if len(c) <= 8]

    # # process cycles to add to HeteroData
    cycle_nodes = []
    cycle_node_to_node = []
    node_to_cycle_node = []
    cycle_node_to_edge_node = []
    edge_node_to_cycle_node = []

    if len(cycles) == 0:
        return hg, mapper

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
        og_cte_to_edge = {}
        og_etc_to_edge = {}

        # connect cycle_node to all nodes in the cycle
        for node in cycle:
            cycle_node_to_node.append([i, node])
            node_to_cycle_node.append([node, i])

        # connect cycle_node to all edges in the cycle
        for j in range(cycle_length):
            edge = (cycle[j], cycle[(j + 1) % cycle_length])
            edge = (min(edge), max(edge))  # as edges are undirected in nx
            # edge_index = edge_list.index(list(edge))
            u = min(edge)
            v = max(edge)
            # u -> multiple edge nodes (edge node -> cycle)
            # ui = node_to_edgenode[0].index(u)
            # edge_index = node_to_edgenode[1][ui]
            # print(edge_index)

            uids = [
                node_to_edgenode[1][idx]
                for idx, u2 in enumerate(node_to_edgenode[0])
                if u2 == u
            ]
            vids = [
                node_to_edgenode[1][idx]
                for idx, v2 in enumerate(node_to_edgenode[0])
                if v2 == v
            ]

            # common id
            edge_index = set(uids).intersection(set(vids)).pop()

            cycle_node_to_edge_node.append([i, edge_index])
            edge_node_to_cycle_node.append([edge_index, i])

            edges_in_cycle.append(edge_index)
            c_t_e_edges.append(c_t_e_counter)
            e_t_c_edges.append(e_t_c_counter)
            # og_cte_to_edge.append(edge_index)
            # og_etc_to_edge.append(edge_index)
            og_cte_to_edge[c_t_e_counter] = edge_index
            og_etc_to_edge[e_t_c_counter] = edge_index

            # mapping[3][c_t_e_counter] = edge_index
            # mapping[4][e_t_c_counter] = edge_index
            mapper["cycle_node_to_edge_node"][i].append(edge_index)

            c_t_e_counter += 1
            e_t_c_counter += 1

    hg["cycle_node"].x = torch.tensor(np.array(cycle_nodes), dtype=torch.float32)
    hg["cycle_node", "to3", "edge_node"].edge_index = torch.tensor(
        np.array(cycle_node_to_edge_node), dtype=torch.long
    ).t()
    hg["edge_node", "to4", "cycle_node"].edge_index = torch.tensor(
        np.array(edge_node_to_cycle_node), dtype=torch.long
    ).t()

    # create edgenode to edgenode connections
    # two edge nodes are connected if they share a common node
    # edge node -> edge node
    edge_conn_cache = defaultdict(bool) # (u, v) -> bool to check if edge node already connected
    edge_node_to_edge_node = ([], [])
    for i in range(len(node_to_edgenode[0]) - 1):
        node_i = node_to_edgenode[0][i]
        edgenode_i = node_to_edgenode[1][i]
        for j in range(i + 1, len(node_to_edgenode[0])):
            node_j = node_to_edgenode[0][j]
            edgenode_j = node_to_edgenode[1][j]
            if  node_i == node_j:
                # check if edge node already connected
                if edge_conn_cache[(min(edgenode_i, edgenode_j), max(edgenode_i, edgenode_j))]:
                    continue
                edge_conn_cache[(min(edgenode_i, edgenode_j), max(edgenode_i, edgenode_j))] = True
                
                edge_node_to_edge_node[0].append(edgenode_i)
                edge_node_to_edge_node[1].append(edgenode_j)
                edge_node_to_edge_node[0].append(edgenode_j)
                edge_node_to_edge_node[1].append(edgenode_i)

    # create cycle node to cycle node connections
    # two cycle nodes are connected if they share a common node
    # cycle node -> cycle node
    cycle_conn_cache = defaultdict(bool) # (u, v) -> bool to check if cycle node already connected
    cycle_node_to_cycle_node = ([], [])
    for i in range(len(node_to_cycle_node) - 1):
        node_i = node_to_cycle_node[i][0]
        cyclenode_i = node_to_cycle_node[i][1]
        for j in range(i + 1, len(node_to_cycle_node)):
            node_j = node_to_cycle_node[j][0]
            cyclenode_j = node_to_cycle_node[j][1]
            if node_i == node_j:
                # check if cycle node already connected
                if cycle_conn_cache[(min(cyclenode_i, cyclenode_j), max(cyclenode_i, cyclenode_j))]:
                    continue
                cycle_conn_cache[(min(cyclenode_i, cyclenode_j), max(cyclenode_i, cyclenode_j))] = True

                cycle_node_to_cycle_node[0].append(cyclenode_i)
                cycle_node_to_cycle_node[1].append(cyclenode_j)
                cycle_node_to_cycle_node[0].append(cyclenode_j)
                cycle_node_to_cycle_node[1].append(cyclenode_i)

    hg["edge_node", "to5", "edge_node"].edge_index = torch.tensor(
        edge_node_to_edge_node, dtype=torch.long
    )
    hg["cycle_node", "to6", "cycle_node"].edge_index = torch.tensor(
        cycle_node_to_cycle_node, dtype=torch.long
    )

    return hg, mapper


class ComplexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_index = dataset.train_index
        self.test_index = dataset.test_index

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

    def get_underlying_graph(self, idx):
        return self.dataset[idx][0]


def lift_dataset(dataset: GraphDataset) -> ComplexDataset:
    """
    Lifts the graphs in the dataset to higher order cell complexes.

    Args:
        dataset (GraphDataset): The graph dataset to lift.

    Returns:
        ComplexDataset: The cell complex dataset.
    """
    return ComplexDataset(dataset)


def load_dataset_as_complex(name="Benzene", seed=0) -> ComplexDataset:
    """
    Loads the dataset and converts it to a cell complex dataset.
    """
    return lift_dataset(load_dataset(name, seed))


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
    train_data = [dataset[i][0] for i in dataset.train_index]
    test_data = [dataset[i][0] for i in dataset.test_index]
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


import matplotlib.pyplot as plt


def visualize_hetero_graph(data, name="hetero_graph"):
    G = nx.DiGraph()

    # Add nodes for each entity type
    node_colors = {}
    for node_type in data.node_types:
        node_attr = data[node_type]
        for i in range(node_attr["x"].shape[0]):
            G.add_node(f"{node_type}_{i}", entity=node_type)
            node_colors[f"{node_type}_{i}"] = node_type

    # Add edges for each relation type
    edge_colors = []
    for edge_type in data.edge_types:
        src, relation, dst = edge_type
        edge_index = data[edge_type].edge_index
        for i in range(edge_index.shape[1]):
            G.add_edge(
                f"{src}_{edge_index[0, i]}",
                f"{dst}_{edge_index[1, i]}",
                relation=relation,
            )
            edge_colors.append(relation)

    # Draw the graph
    pos = nx.spring_layout(G, k=0.3)
    unique_node_types = set(node_colors.values())
    unique_relations = set(edge_colors)

    color_map = {
        etype: plt.cm.tab20(i / len(unique_node_types))
        for i, etype in enumerate(unique_node_types)
    }
    edge_color_map = {
        rel: plt.cm.tab20(i / len(unique_relations))
        for i, rel in enumerate(unique_relations)
    }

    for etype in unique_node_types:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[
                node for node, attr in G.nodes(data=True) if attr["entity"] == etype
            ],
            node_color=[color_map[etype]],
            label=etype,
            node_size=50,
        )

    for rel in unique_relations:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[
                (u, v) for u, v, attr in G.edges(data=True) if attr["relation"] == rel
            ],
            edge_color=[edge_color_map[rel]],
            label=rel,
        )

    # print edge_color_map legend
    for i, rel in enumerate(unique_relations):
        plt.plot([0], [0], color=edge_color_map[rel], label=rel)

    # convert to graphml
    nx.write_graphml(G, f"{name}.graphml")

    plt.legend()
    # plt.show()
    plt.savefig("hetero_graph.png")

def pyg_to_graphml(pyg_data, output_file):
    # Convert PyG data to NetworkX graph
    G = to_networkx(pyg_data, to_undirected=True)
    
    # Add node features as attributes
    for i, features in enumerate(pyg_data.x):
        for j, feature in enumerate(features):
            G.nodes[i][f'feature_{j}'] = feature.item()
    
    # Add edge features as attributes (if available)
    if pyg_data.edge_attr is not None:
        for i, (u, v) in enumerate(pyg_data.edge_index.t()):
            for j, feature in enumerate(pyg_data.edge_attr[i]):
                G.edges[u.item(), v.item()][f'feature_{j}'] = feature.item()
    
    # Write to GraphML file
    nx.write_graphml(G, output_file)

if __name__ == "__main__":
    args.synth_shape_1 = "cycle_6"
    args.synth_shape_2 = "house"
    dataset = load_dataset(name="Synth", seed=34)

    # convert pyg data object to graphml
    for i in range(len(dataset)):
        data = dataset[i][0]
        print(data, data.y, data.y.item())
        if data.y.item() == 1:
            print(data)
            pyg_to_graphml(data, f"synth_{i}_class_0.graphml")
            break


    # graph, expl = dataset[0]
    # print(graph)
    # complex, _ = graph_to_complex(graph)
    # full_complex, mapping = graph_to_full_complex(graph)
    # print(complex)
    # print(mapping)

    # visualize_hetero_graph(complex)
    # visualize_hetero_graph(full_complex, name="full_hetero_graph")
