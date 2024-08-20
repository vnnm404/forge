import os
import subprocess

def execute_shell_scripts(directory):
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'gnn' not in file:
                continue
            if file.endswith(".sh"):
                script_path = os.path.join(root, file)
                print(f"Running {script_path}")
                # Execute the shell script
                os.system(f"sh {script_path}")

if __name__ == "__main__":
    # Define the directory containing the shell scripts
    directory = "experiments/graph_classification/synthetic"
    
    # Execute all shell scripts in the directory and subdirectories
    execute_shell_scripts(directory)
