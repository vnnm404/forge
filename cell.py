from graphxai.datasets import Benzene

import dgl
from dgl.dataloading import GraphDataLoader
from dgl.data import TUDataset
from dgl.nn import GraphConv, GATConv, RelGraphConv, HeteroGraphConv
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.pytorch import HeteroGNNExplainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

import matplotlib.pyplot as plt

import tyro
from sklearn.model_selection import KFold


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)


# @dataclass
# class Args:
#     dataset: str = 'Mutagenicity'
#     node_feat: int = 14
#     edge_feat: int = 3
#     cycle_feat: int = 8

@dataclass
class Args:
    dataset: str = 'Benzene'
    node_feat: int = 14
    edge_feat: int = 5
    cycle_feat: int = 8

args = tyro.cli(Args)


def to_complex(g):
    data_dict = {
        ('node', 'to_0', 'node'): g.edges(),
    }

    node_u, node_v, edge_id = g.edges(form='all')
    cache = defaultdict(list)
    mapper = defaultdict(lambda: defaultdict(list))

    # edges
    node_to_edgenode = ([], [])
    edgenode_to_node = ([], [])
    edge_feat = []
    edgenode_id = 0
    for i in range(len(node_u)):
        u = node_u[i].item()
        v = node_v[i].item()
        id = edge_id[i].item()
        feat = g.edata['edge_labels'][id].item()
        if (min(u, v), max(u, v)) in cache:
            mapper['edgenode'][cache[(min(u, v), max(u, v))]].append(id)
            continue

        cache[(min(u, v), max(u, v))] = edgenode_id
        mapper['edgenode'][edgenode_id].append(id)
        
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
    
    edge_feat = torch.tensor(edge_feat)
    data_dict[('node', 'to_1', 'edgenode')] = (torch.tensor(node_to_edgenode[0]), torch.tensor(node_to_edgenode[1]))
    data_dict[('edgenode', 'to_2', 'node')] = (torch.tensor(edgenode_to_node[0]), torch.tensor(edgenode_to_node[1]))

    # cycles
    nx_graph = nx.Graph()
    for i in range(len(node_u)):
        u = node_u[i].item()
        v = node_v[i].item()
        nx_graph.add_edge(u, v)

    cycles = [cycle for cycle in nx.simple_cycles(nx_graph) if 3 <= len(cycle) <= args.cycle_feat]
    cyclenode_to_edgenode = ([], [])
    edgenode_to_cyclenode = ([], [])
    cycle_feat = []
    cyclenode_id = 0
    if cycles:
        for cycle in cycles:
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                edge_id = cache[(min(u, v), max(u, v))]

                cyclenode_to_edgenode[0].append(cyclenode_id)
                cyclenode_to_edgenode[1].append(edge_id)
                edgenode_to_cyclenode[0].append(edge_id)
                edgenode_to_cyclenode[1].append(cyclenode_id)

                mapper['cyclenode'][cyclenode_id].extend(mapper['edgenode'][edge_id])

            cycle_feat.append(len(cycle))
            cyclenode_id += 1
    else:
        cyclenode_id += 1
        cycle_feat.append(0)
        cyclenode_to_edgenode[0].append(0)
        cyclenode_to_edgenode[1].append(0)
        edgenode_to_cyclenode[0].append(0)
        edgenode_to_cyclenode[1].append(0)

    cycle_feat = torch.tensor(cycle_feat)
    data_dict[('cyclenode', 'to_3', 'edgenode')] = (torch.tensor(cyclenode_to_edgenode[0]), torch.tensor(cyclenode_to_edgenode[1]))
    data_dict[('edgenode', 'to_4', 'cyclenode')] = (torch.tensor(edgenode_to_cyclenode[0]), torch.tensor(edgenode_to_cyclenode[1]))

    # make
    num_nodes_dict = {
        'node': len(g.nodes()),
        'edgenode': edgenode_id,
        'cyclenode': cyclenode_id
    }

    hg = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    hg.nodes['node'].data['feat'] = F.one_hot(g.ndata['node_labels'].squeeze(dim=-1), num_classes=args.node_feat).float()
    hg.nodes['edgenode'].data['feat'] = F.one_hot(edge_feat, num_classes=args.edge_feat).float()

    if cycle_feat[0].item() == 0:
        hg.nodes['cyclenode'].data['feat'] = torch.zeros(1, args.cycle_feat)
    else:
        hg.nodes['cyclenode'].data['feat'] = F.one_hot(cycle_feat - 1, num_classes=args.cycle_feat).float()

    return hg, mapper


def plot_graph(g, edge_mask=None):
    g_nx = g.to_networkx(node_attrs=None, edge_attrs=None)

    edge_colors = ['black'] * len(g_nx.edges())
    if edge_mask is not None:
        edge_index_map = {edge: idx for idx, edge in enumerate(g_nx.edges())}

        for idx, (u, v) in enumerate(g_nx.edges()):
            if edge_mask[idx]:
                edge_colors[edge_index_map[(u, v)]] = 'green'
                edge_colors[idx] = 'green'
    
    pos = nx.spring_layout(g_nx)
    nx.draw(g_nx, pos, with_labels=True, edge_color=edge_colors, node_size=300)
    plt.title('plot')
    plt.axis('off')
    plt.show()


def plot_complex(g):
    # Convert DGL Graph to NetworkX Graph for visualization
    g_nx = g.to_networkx(node_attrs=None, edge_attrs=None)
    # print(g_nx.nodes(data=True))
    # print(g_nx.edges(data=True))

    color_map = []
    for node in g_nx:
        if g_nx.nodes[node]['ntype'] == 'edgenode':
            color_map.append('lightblue')
        elif g_nx.nodes[node]['ntype'] == 'cyclenode':
            color_map.append('green')
        else:
            color_map.append('red')

    pos = nx.spring_layout(g_nx)
    nx.draw(g_nx, pos, node_color=color_map, with_labels=True, node_size=300)
    plt.title('plot')
    plt.axis('off')
    plt.show()


class GNN(nn.Module):
    def __init__(self, node_in_feats, edge_feat_dim, num_heads=2, dropout=0.1):
        super(GNN, self).__init__()
        self.e_dim = 32
        self.h_dim = 16
        self.num_heads = num_heads
        self.dropout = dropout

        self.feature_transform = nn.ModuleDict({
            'node': nn.Linear(node_in_feats, self.e_dim),
            'edgenode': nn.Linear(edge_feat_dim, self.e_dim),
            'cyclenode': nn.Linear(args.cycle_feat, self.e_dim),
        })

        self.layer1 = HeteroGraphConv({
            'to_0': GATConv(self.e_dim, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
            'to_1': GATConv(self.e_dim, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
            'to_2': GATConv(self.e_dim, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
            'to_3': GATConv(self.e_dim, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
            'to_4': GATConv(self.e_dim, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
        }, aggregate='mean')

        self.layer2 = HeteroGraphConv({
            'to_0': GATConv(self.h_dim * num_heads, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
            'to_1': GATConv(self.h_dim * num_heads, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
            'to_2': GATConv(self.h_dim * num_heads, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
            'to_3': GATConv(self.h_dim * num_heads, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
            'to_4': GATConv(self.h_dim * num_heads, self.h_dim, num_heads=num_heads, allow_zero_in_degree=True),
        }, aggregate='mean')

        self.batchnorm1 = nn.BatchNorm1d(self.h_dim * num_heads)
        self.batchnorm2 = nn.BatchNorm1d(self.h_dim * num_heads)

        self.out = nn.Linear(self.h_dim * num_heads * 3, 1)
        # self.out = nn.Linear(self.h_dim * num_heads, 1)

    def forward(self, graph, feat, eweight=None):
        h_dict = {ntype: self.feature_transform[ntype](feat[ntype]) for ntype in graph.ntypes}
        
        h_dict = self.layer1(graph, h_dict)
        h_dict = {k: F.relu(v.view(-1, self.h_dim * self.num_heads)) for k, v in h_dict.items()}
        h_dict = {k: self.batchnorm1(v) for k, v in h_dict.items()}
        h_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h_dict.items()}

        h_dict = self.layer2(graph, h_dict)
        h_dict = {k: F.relu(v.view(-1, self.h_dim * self.num_heads)) for k, v in h_dict.items()}
        h_dict = {k: self.batchnorm2(v) for k, v in h_dict.items()}
        h_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h_dict.items()}

        h = []
        with graph.local_scope():
            for ntype in graph.ntypes:
                graph.nodes[ntype].data['h'] = h_dict[ntype]
                pool_h = dgl.mean_nodes(graph, 'h', ntype=ntype)
                h.append(pool_h)
            
        h = torch.cat(h, dim=1)
        h = self.out(h)

        # h = 0
        # with graph.local_scope():
        #     for ntype in graph.ntypes:
        #         graph.nodes[ntype].data['h'] = h_dict[ntype]
        #         h += dgl.mean_nodes(graph, 'h', ntype=ntype)

        # h = self.out(h)
        return h


def train_step(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for hg, labels in dataloader:
        feat = hg.ndata['feat']
        pred = model(hg, feat)
        loss = F.binary_cross_entropy_with_logits(pred, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_step(model, dataloader):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for hg, labels in dataloader:
            feat = hg.ndata['feat']
            pred = model(hg, feat)
            loss = F.binary_cross_entropy_with_logits(pred, labels.float())
            total_loss += loss.item()
            total_acc += ((pred > 0) == labels).sum().item()
    return total_loss / len(dataloader), total_acc / len(dataloader.dataset)


def train(model, train_dataloader, val_dataloader, optimizer, n_epochs):
    best_val_loss = float('inf')
    best_val_accuracy = 0
    for epoch in tqdm(range(n_epochs), desc="train", leave=False):
        train_loss = train_step(model, train_dataloader, optimizer)
        val_loss, val_acc = eval_step(model, val_dataloader)

        new_best = False
        if val_loss < best_val_loss:
            new_best = True
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'models/cell.pth')
        
        print(f"{'!' if new_best else ''}[{epoch}] tr {train_loss:.3f} | val {val_loss:.3f} | val_acc {val_acc:.3f}")
    
    print(f'best val {best_val_loss} | acc {best_val_accuracy}')


'''Benzene Dataset Compatability Stuff'''


def pyg_to_dgl(data):
    src, dst = data.edge_index[0], data.edge_index[1]
    g = dgl.graph((src, dst))

    if 'x' in data:
        g.ndata['node_labels'] = data.x.argmax(dim=-1)

    if 'edge_attr' in data:
        g.edata['edge_labels'] = data.edge_attr.argmax(dim=-1)

    return g


class BenzeneDGL(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = Benzene()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        g, explanations = self.dataset[idx]
        dgl_graph = pyg_to_dgl(g)
        label = g.y
        hg, mapping = to_complex(dgl_graph)

        return hg, label, mapping, explanations, dgl_graph
    
    def collate_fn(self, batch):
        graphs, labels, mappings, explanations, _ = zip(*batch)
        hg = dgl.batch(graphs)
        labels = torch.tensor(labels)
        return hg, labels.unsqueeze(dim=-1)


if __name__ == '__main__':
    # dataset = TUDataset(args.dataset, verbose=False)
    # g = dataset[0][0]
    # hg, mapper = to_complex(g)
    # print(mapper)
    # print(hg)
    # plot_graph(g)
    # plot_complex(hg)

    lifted_dataset = BenzeneDGL()
    g = lifted_dataset[0][4]
    hg = lifted_dataset[0][0]
    print(g)
    # plot_complex(hg)

    # train_dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)
    # for hg, label, mapping, explanations in train_dataloader:
    #     print(hg)
    #     print(label)
    #     print(mapping)
    #     print(explanations)
    #     break

    # lifted_dataset = [(to_complex(g)[0], label) for g, label in tqdm(dataset, desc="to_complex")]
    # k = 6
    # kf = KFold(n_splits=k, shuffle=True)

    # val_losses = []
    # val_accuracies = []

    # for train_index, val_index in tqdm(kf.split(lifted_dataset), total=k, desc="folds"):
    #     train_set = [lifted_dataset[i] for i in train_index]
    #     val_set = [lifted_dataset[i] for i in val_index]

    #     train_dataloader = GraphDataLoader(train_set, batch_size=32, shuffle=True, collate_fn=lifted_dataset.collate_fn)
    #     val_dataloader = GraphDataLoader(val_set, batch_size=32, shuffle=False, collate_fn=lifted_dataset.collate_fn)

    #     model = GNN(args.node_feat, args.edge_feat)
    #     optimizer = optim.Adam(model.parameters(), lr=0.01)
        
    #     train(model, train_dataloader, val_dataloader, optimizer, 10)

    #     model.load_state_dict(torch.load('models/cell.pth'))
        
    #     val_loss, val_acc = eval_step(model, val_dataloader)
        
    #     val_losses.append(val_loss)
    #     val_accuracies.append(val_acc)

    # avg_val_loss = sum(val_losses) / k
    # avg_val_accuracy = sum(val_accuracies) / k

    # print(f'Average validation loss: {avg_val_loss:.3f}')
    # print(f'Average validation accuracy: {avg_val_accuracy:.3f}')
    # print(f'Best validation loss: {min(val_losses):.3f}')
    # print(f'Best validation accuracy: {max(val_accuracies):.3f}')

    model = GNN(args.node_feat, args.edge_feat)
    model.load_state_dict(torch.load('models/cell.pth'))

    # generate explanation for hg
    explainer = HeteroGNNExplainer(model, num_hops=2, num_epochs=100, lr=0.01)
    
    feat = hg.ndata['feat']
    feat_mask, edge_mask = explainer.explain_graph(hg, feat)
    # print(edge_mask[('node', 'to_0', 'node')] > 0.5)

    # plot explanation
    plot_graph(g, edge_mask[('node', 'to_0', 'node')] > 0.5)
