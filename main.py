from config import args
from graph_classification import graph_classification


if __name__ == "__main__":
    print("============= Args =============")
    print(args)
    print("================================")
    if args.task_level == "graph":
        graph_classification()
    else:
        raise NotImplementedError(f"Task level {args.task_level} is not implemented.")