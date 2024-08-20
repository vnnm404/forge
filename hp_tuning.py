import os
import subprocess

def execute_shell_scripts(directory):
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "graph" in file or "random" in file:
                continue
            if "benzene" in file:
                if "gnn" in file or "grad" in file or "_gme" in file:
                    if file.endswith(".sh"):
                        script_path = os.path.join(root, file)
                        print(f"Running {script_path}")
                        # Execute the shell script
                        os.system(f"sh {script_path}")

if __name__ == "__main__":
    # Define the directory containing the shell scripts
    directory = "experiments/graph_classification/"
    
    # Execute all shell scripts in the directory and subdirectories
    execute_shell_scripts(directory)
