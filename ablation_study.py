from functools import partial
import json
import torch 
import numpy as np
import pickle
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm(x):
    # x is an np array
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def hierarchical_prop(edge_mask, mappings, alpha_c=1.0, alpha_e=1.0):
    og_edge_set = set()
    for edge_pair in mappings["edge_node_to_edge"]:
        edge_0_0 = edge_pair[1]
        og_edge_set.add(edge_0_0)
        
    num_og_edges = len(og_edge_set)
    
    new_edge_mask = np.zeros(num_og_edges)
    cycle_node_to_edge_node = mappings["cycle_node_to_edge_node"]
    edge_node_to_edge = mappings["edge_node_to_edge"]

    for edge_pair in cycle_node_to_edge_node:
        edge_1_2 = edge_pair[0]
        edge_0_1 = edge_pair[1]
        edge_mask[edge_0_1] += (edge_mask[edge_1_2] - 0.5) * alpha_c

    for edge_pair in edge_node_to_edge:
        edge_0_1 = edge_pair[0]
        edge_0_0 = edge_pair[1]
        edge_mask[edge_0_0] += (edge_mask[edge_0_1] - 0.5) * alpha_e

    new_edge_mask[:num_og_edges] = edge_mask[:num_og_edges]

    return new_edge_mask

dir = "hp_tuning"

preds = []
gts = []
maps = []

for i in range(100):
    preds.append(np.load(f"{dir}/complex_explanations_{i}.npy"))
    gts.append(np.load(f"{dir}/complex_ground_truth_{i}.npy"))
    maps.append(pickle.load(open(f"{dir}/mappings_{i}.pkl", "rb")))

def run_hp(alpha_c, alpha_e):
    new_preds = []
    for i in range(100):
        new_preds.append(norm(hierarchical_prop(preds[i], maps[i], alpha_c, alpha_e)))
        
    # get jaccard scores
    jaccard_scores = []

    for i in range(100):
        # threshold predictions
        new_preds[i] = new_preds[i] > 0.5
        jaccard_scores.append(jaccard_score(gts[i], new_preds[i]))
    # average jaccard score
    avg_jaccard_score = sum(jaccard_scores) / len(jaccard_scores)
    
    return avg_jaccard_score

def get_random_float(a,b):
    # get a random float between a and b
    return a + (b-a) * np.random.rand()

vals = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]

def objective():
    res = []
    for i in tqdm(range(len(vals) - 1)):
        alpha_c = get_random_float(vals[i], vals[i+1])
        for j in range(len(vals)-1):
            if i==j:
                continue
            alpha_e = get_random_float(vals[j], vals[j+1])
            
            jacc = run_hp(alpha_c, alpha_e)
            
            res.append({
                'alpha_c': alpha_c,
                'alpha_e': alpha_e,
                'jacc': jacc
            })
            
            print(f"alpha_c: {alpha_c}, alpha_e: {alpha_e}, jacc: {jacc}")
            
    # save results
    with open("alpha_ablation.json", "w") as f:
        json.dump(res, f)


objective()

# plot results
with open("alpha_ablation.json", "r") as f:
    data = json.load(f)
    
    x = []
    y = []
    # color by jaccard score
    value = []
    for d in data:
        x.append(d['alpha_c'])
        y.append(d['alpha_e'])
        value.append(d['jacc'])
    

    # Set the style
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(8.2, 8.2))
    plt.tight_layout()
    # Use a sequential colormap 
    colors = sns.color_palette("viridis", as_cmap=True)
    
    # Plot the points
    plt.scatter(x, y, s=value)
    
    # save plot
    plt.savefig("alpha_ablation.png")
    plt.show()