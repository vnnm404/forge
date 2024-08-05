import os


dir = "experiments/graph_classification"

# get all subfolders in the directory
subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]

# get the subsubfolders
subsubfolders = []
for subfolder in subfolders:
    subsubfolders += [f.path for f in os.scandir(subfolder) if f.is_dir()]
    
print(subsubfolders)

# the name of the subsubfolder is the dataset name

folder_to_dataset = {
    'house_X_wheel'
}