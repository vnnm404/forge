from graphxai.datasets import (
    Benzene,
    AlkaneCarbonyl,
    GraphDataset,
    Mutagenicity,
    FluorideCarbonyl,
)
import csv
from tqdm import tqdm
import torch
from graphxai.datasets.synth import Synth
from torch_geometric.data import HeteroData
import networkx as nx
import numpy as np


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
        return hg, mapping, 0, 0

    max_cycle_length = max([len(c) for c in cycles])

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

    return hg, mapping, len(cycles), max_cycle_length


def load_dataset(name="Benzene"):
    if name == "Benzene":
        return Benzene()
    elif name == "AlkaneCarbonyl":
        return AlkaneCarbonyl()
    elif name == "Mutagenicity":
        return Mutagenicity(root="data/")
    elif name == "FluorideCarbonyl":
        return FluorideCarbonyl()
    elif name == "House/Wheel":
        return Synth(
            num_samples=2000,
            shape1="house",
            shape2="wheel",
        )
    elif name == "Bull/Square":
        return Synth(
            num_samples=2000,
            shape1="bull",
            shape2="cycle_4",
        )
    elif name == "Wheel/Cube":
        return Synth(
            num_samples=2000,
            shape1="wheel",
            shape2="cube",
        )
    elif name == "House/Hex":
        return Synth(
            num_samples=2000,
            shape1="house",
            shape2="cycle_6",
        )
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")


def get_data_stats(dataset):
    n_nodes = 0
    n_edges = 0
    n_graphs = 0
    n_cycles = 0

    for i in tqdm(range(len(dataset))):
        data, _, num_cycles, max_cycle_length = graph_to_complex(dataset[i][0])
        data = data.to_homogeneous()
        n_nodes += data.num_nodes
        n_edges += data.num_edges
        n_cycles += num_cycles
        n_graphs += 1

    avg_nodes = n_nodes / n_graphs
    avg_edges = n_edges / n_graphs
    avg_num_cycles = n_cycles / n_graphs

    avg_degree = avg_edges / avg_nodes

    print(
        f"""Average number of nodes: {avg_nodes}\n
        Average number of edges: {avg_edges}\n
        Average degree: {avg_degree}\n
        Number of graphs: {n_graphs}\n
        Avg Number of cycles: {avg_num_cycles}\n
        Maximum cycle length: {max_cycle_length}\n""",
        end="\r",
    )

    return avg_nodes, avg_edges, avg_degree, n_graphs, avg_num_cycles, max_cycle_length


def save_to_csv(dataset_names):
    with open("data_stats.csv", mode="w") as data_stats:
        data_stats_writer = csv.writer(
            data_stats, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        data_stats_writer.writerow(
            [
                "Dataset",
                "Avg Nodes",
                "Avg Edges",
                "Avg Degree",
                "Num Cycles",
                "Max Cycle Length",
                "Num Graphs",
            ]
        )

        for name in dataset_names:
            dataset = load_dataset(name)
            print(f"Calculating stats for {name}...")
            avg_nodes, avg_edges, avg_degree, n_graphs, num_cycles, max_cycle_length = (
                get_data_stats(dataset)
            )
            data_stats_writer.writerow(
                [
                    name,
                    avg_nodes,
                    avg_edges,
                    avg_degree,
                    num_cycles,
                    max_cycle_length,
                    n_graphs,
                ]
            )


if __name__ == "__main__":
    dataset_names = ["Benzene", "AlkaneCarbonyl", "Mutagenicity", "FluorideCarbonyl", "House/Wheel", "Bull/Square", "Wheel/Cube", "House/Hex"]
    save_to_csv(dataset_names)
