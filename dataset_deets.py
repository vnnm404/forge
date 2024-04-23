from graphxai.datasets import Benzene, AlkaneCarbonyl, GraphDataset, Mutagenicity, FluorideCarbonyl
import csv
from tqdm import tqdm
import torch

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

def get_data_stats(dataset):
    n_nodes = 0
    n_edges = 0
    n_graphs = 0
    
    for i in tqdm(range(len(dataset))):
        data = dataset[i][0]
        n_nodes += data.num_nodes
        n_edges += data.num_edges
        n_graphs += 1
    
    avg_nodes = n_nodes / n_graphs
    avg_edges = n_edges / n_graphs
    
    avg_degree = avg_edges / avg_nodes
    
    return avg_nodes, avg_edges, avg_degree, n_graphs

def save_to_csv(dataset_names):
    with open('data_stats.csv', mode='w') as data_stats:
        data_stats_writer = csv.writer(data_stats, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        data_stats_writer.writerow(['Dataset', 'Avg Nodes', 'Avg Edges', 'Avg Degree', 'Num Graphs'])
        
        for name in dataset_names:
            dataset = load_dataset(name)
            print(f'Calculating stats for {name}...')
            avg_nodes, avg_edges, avg_degree, n_graphs = get_data_stats(dataset)
            data_stats_writer.writerow([name, avg_nodes, avg_edges, avg_degree, n_graphs])

if __name__ == '__main__':
    # dataset_names = ['Benzene', 'AlkaneCarbonyl', 'Mutagenicity', 'FluorideCarbonyl']
    # save_to_csv(dataset_names)
    dataset = torch.load("data/SG-SmallEx.pt")
    print(dataset)