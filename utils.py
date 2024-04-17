from explain_utils import visualise_explanation, initialise_explainer, explain_dataset
from config import args

def visualise(model, dataset, num=50):
    explainer = initialise_explainer(
        model=model,
        explanation_algorithm_name=args.explanation_algorithm,
        explanation_epochs=args.explanation_epochs,
        explanation_lr=args.explanation_lr,
    )

    pred_explanations, ground_truth_explanations = explain_dataset(
        explainer, dataset, num=num
    )