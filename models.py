import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
# data loader
from torch_geometric.data import DataLoader, Data

class GCN(torch.nn.Module):
    def __init__(self, in_dim=None, hidden_dim=None, out_dim=None):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch) 
        x = self.lin(x)
        # sigmoid
        x = F.sigmoid(x)
        x = x.squeeze(1)
        return x