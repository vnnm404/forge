from config import args
from graph_classification import graph_classification


if __name__ == "__main__":
    if args.task_level == "graph":
        graph_classification()