# Databricks notebook
import mlflow

experiment_id = mlflow.get_experiment_by_name("/Shared/ExampleExperiment")["experiment_id"]
client = mlflow.tracking.MlflowClient()

runs = client.search_runs(experiment_ids=[experiment_id], order_by=["metrics.accuracy DESC"])
best_run = runs[0]

result = mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/model",
    name="example_model_registry"
)

print(f"Modelo registrado: {result.name} v{result.version}")
