import dgl
from dgl.dataloading import GraphDataLoader
from dgl.data import TUDataset
from dgl.nn import GraphConv, GATConv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)


class GNN(nn.Module):
    def __init__(self, in_dim, h_dim, n_classes, num_heads=4, dropout=0.05):
        super(GNN, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.n_classes = n_classes
        self.num_heads = num_heads
        self.dropout = dropout

        self.initial_embed = nn.Linear(in_dim, h_dim)

        self.conv1 = GATConv(h_dim, h_dim, num_heads=num_heads, allow_zero_in_degree=True)
        self.conv2 = GATConv(h_dim * num_heads, h_dim, num_heads=num_heads, allow_zero_in_degree=True)

        self.batchnorm1 = nn.BatchNorm1d(h_dim * num_heads)
        self.batchnorm2 = nn.BatchNorm1d(h_dim * num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim * num_heads, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, n_classes)
        )

    def forward(self, g, in_feat):
        in_feat = F.one_hot(in_feat.squeeze(dim=-1), num_classes=self.in_dim).float()
        in_feat = self.initial_embed(in_feat)

        h = F.relu(self.conv1(g, in_feat).view(-1, self.h_dim * self.num_heads))
        h = self.batchnorm1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.relu(self.conv2(g, h).view(-1, self.h_dim * self.num_heads))
        h = self.batchnorm2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
        
        out = self.mlp(hg)
        return out


def train_step(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for g, label in dataloader:
        pred = model(g, g.ndata['node_labels'])
        loss = F.binary_cross_entropy_with_logits(pred, label.float())
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
        for g, label in dataloader:
            pred = model(g, g.ndata['node_labels'])
            loss = F.binary_cross_entropy_with_logits(pred, label.float())
            total_loss += loss.item()
            total_acc += ((pred > 0) == label).sum().item()
    return total_loss / len(dataloader), total_acc / len(dataloader.dataset)


def train(model, train_dataloader, val_dataloader, optimizer, n_epochs):
    best_val_loss = float('inf')
    best_val_accuracy = 0
    for epoch in range(n_epochs):
        train_loss = train_step(model, train_dataloader, optimizer)
        val_loss, val_acc = eval_step(model, val_dataloader)

        new_best = False
        if val_loss < best_val_loss:
            new_best = True
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'models/base.pth')
        
        print(f"{'!' if new_best else ''}[{epoch}] tr {train_loss:.3f} | val {val_loss:.3f} | acc {val_acc:.3f}")
    
    print(f'best val {best_val_loss} | acc {best_val_accuracy}')


if __name__ == '__main__':
    dataset = TUDataset('Mutagenicity', verbose=False)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    train_dataloader = GraphDataLoader(train_set, batch_size=32, shuffle=True)
    val_dataloader = GraphDataLoader(val_set, batch_size=32, shuffle=False)
    test_dataloader = GraphDataLoader(test_set, batch_size=32, shuffle=False)

    model = GNN(14, 64, 1)
    # optimizer = optim.Adam(model.parameters(), lr=0.005)
    # train(model, train_dataloader, val_dataloader, optimizer, 25)

    model.load_state_dict(torch.load('models/base.pth'))
    
    test_loss, test_acc = eval_step(model, test_dataloader)
    print(f'ts {test_loss:.3f} | acc {test_acc:.3f}')
