# %%
from data import load_dataset, load_dataset_as_complex, graph_to_complex
from time import time

# %%
st = time()
dataset = load_dataset(name='Benzene')
print('Dataset loaded in', time() - st, 's')

st = time()
complex_dataset = load_dataset_as_complex()
print('Dataset loaded in', time() - st, 's')

# %%
for data, exp in dataset:
    print(data)
    break

# %%
for data, exp, mapping in complex_dataset:
    print(mapping)
    break

# %%
st = time()
for data, exp in dataset:
    graph_to_complex(data)
et = time()
print('Graph to complex in', et - st, 's')
print('Graph to complex in', (et - st) / len(dataset), 's per graph')

# %%
from data import get_data_loaders


train_loader, test_loader = get_data_loaders(dataset, batch_size=128)
complex_train_loader, complex_test_loader = get_data_loaders(complex_dataset, batch_size=128)

# %%
for data in train_loader:
    print(data)
    break

# %%
for data in complex_train_loader:
    print(data)
    break

# %%
import os

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool

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

# %%

avg_time = 0
for i in range(10):
    model = GCN(in_dim=14, hidden_dim=32, out_dim=1)

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    st = time()
    for epoch in range(50):
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.binary_cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
        print(epoch, loss, end='\r')
    et = time()
    print('Training in', et - st, 's')
    avg_time += et-st
avg_time /= 10
print('Average training time:', avg_time)

# %%
avg_time = 0
for i in range(10):
    complex_model = GCN(in_dim=14, hidden_dim=32, out_dim=1)

    # train
    optimizer = torch.optim.Adam(complex_model.parameters(), lr=0.01, weight_decay=5e-4)
    complex_model.train()
    st = time()
    for epoch in range(50):
        for data in complex_train_loader:
            optimizer.zero_grad()
            out = complex_model(data.x, data.edge_index, data.batch)
            loss = F.binary_cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
        print(epoch, loss, end='\r')
    et = time()
    print('Training in', et - st, 's')
    avg_time += et - st
print('Average training time:', avg_time/10)

# %%
from explain_utils import initialise_explainer, explain_dataset

avg_time = 0
for i in range(10):
    graph_explainer = initialise_explainer(model=model, explanation_algorithm_name='GNNExplainer')

    st = time()
    res = explain_dataset(graph_explainer, dataset)
    et = time()
    print("time to explain graphs:", et - st)
    avg_time += et - st
print("Avg graph explaination time:", avg_time/10)

# %%
avg_time = 0
for i in range(10):
    complex_explainer = initialise_explainer(model=complex_model, explanation_algorithm_name="GNNExplainer")

    st = time()
    res = explain_dataset(complex_explainer, complex_dataset)
    et = time()
    print('Time to explain complexes:', et - st)
    avg_time += et-st
print("Avg complex explaination time:", avg_time/10)
