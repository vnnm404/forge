import torch
from torch_geometric.nn import GCNConv, GATConv


def load_model(name="GCN", in_dim=14, hidden_dim=64, out_dim=2):
    if name == "GCN":
        return GCN(in_dim, hidden_dim, out_dim)
    elif name == "GAT":
        return GAT(in_dim, hidden_dim, out_dim)
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")

class GCN(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels, classes = 2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, input_feat, hidden_channels, classes = 2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_feat, hidden_channels)
        self.conv2 = GATConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x