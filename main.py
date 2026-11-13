"""Training entry point for the customer churn project.

Runs the end-to-end pipeline (ingestion, preprocessing, training,
evaluation) and prints the headline test metrics.
"""

from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    metrics = TrainPipeline().train()
    churn = metrics["report"]["churn"]
    print(
        f"\nTest ROC-AUC: {metrics['roc_auc']}  PR-AUC: {metrics['pr_auc']}  "
        f"Churn recall: {churn['recall']:.2f}  Churn precision: {churn['precision']:.2f}"
    )
