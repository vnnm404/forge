import os
import random
import torch

from graphxai.datasets.real_world.extract_google_datasets import load_graphs
from graphxai.datasets import GraphDataset


ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "Na", "Ca", "I", "B", "H", "*"]

fc_data_dir = os.path.join(os.path.dirname(__file__), "fc_data")
fc_smiles_df = "FC_smiles.csv"

fc_datapath = os.path.join(os.path.dirname(__file__), "fluoride_carbonyl.npz")


class FluorideCarbonyl(GraphDataset):

    def __init__(
        self,
        split_sizes=(0.8, 0.2, 0),
        seed=None,
        data_path: str = fc_datapath,
        downsample=True,
        device=None,
    ):
        """
        Args:
            split_sizes (tuple):
            seed (int, optional):
            data_path (str, optional):
        """

        self.device = device
        self.graphs, self.explanations, self.zinc_ids = load_graphs(data_path)
        
        # Downsample because of extreme imbalance:
        yvals = [self.graphs[i].y for i in range(len(self.graphs))]

        zero_bin = []
        one_bin = []

        if downsample:
            for i in range(len(self.graphs)):
                if self.graphs[i].y == 0:
                    zero_bin.append(i)
                else:
                    one_bin.append(i)

            # Sample down to keep the dataset balanced
            random.seed(seed)
            keep_inds = random.sample(zero_bin, k=1 * len(one_bin))
            
            # randomly permute the indices
            indices = keep_inds + one_bin
            random.shuffle(indices)
            print("class 0: ", len(keep_inds))
            print("class 1: ", len(one_bin))

            self.graphs = [self.graphs[i] for i in indices]
            self.explanations = [self.explanations[i] for i in indices]
            self.zinc_ids = [self.zinc_ids[i] for i in indices]

        # self.graphs, self.explanations, self.zinc_ids = \
        #     load_graphs(data_path, os.path.join(data_path, fc_smiles_df))


        super().__init__(
            name="FluorideCarbonyl", seed=seed, split_sizes=split_sizes, device=device
        )
