import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader, Data
from config import args

def load_model(name="GCN", in_dim=14, hidden_dim=64, out_dim=1):
    if name == "GCN":
        return GCN(in_dim, hidden_dim, out_dim)
    elif name == "GAT":
        return GAT(in_dim, hidden_dim, out_dim)
    elif name == "GIN":
        return GIN(in_dim, hidden_dim, out_dim)
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")

class GCN(torch.nn.Module):
    def __init__(self, in_dim=None, hidden_dim=None, out_dim=None):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        # sigmoid
        x = F.sigmoid(x)
        x = x.squeeze(1)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_dim=None, hidden_dim=None, out_dim=None):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        # sigmoid
        x = F.sigmoid(x)
        x = x.squeeze(1)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_dim=None, hidden_dim=None, out_dim=None):
        super(GIN, self).__init__()
        self.mlp1 = torch.nn.Linear(in_dim, hidden_dim)
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = GINConv(self.mlp2)
        self.mlp3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = GINConv(self.mlp3)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = F.sigmoid(x)
        x = x.squeeze(1)
        return x
