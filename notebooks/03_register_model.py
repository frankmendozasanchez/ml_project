import os
import mlflow

experiment_path = os.getenv("DATABRICKS_EXPERIMENT_PATH")
model_name = os.getenv("DATABRICKS_MODEL_NAME")

experiment = mlflow.get_experiment_by_name(experiment_path)
if experiment is None:
    raise ValueError(f"El experimento '{experiment_path}' no existe")

client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])

if not runs:
    raise ValueError("No hay ejecuciones registradas")

best_run = runs[0]

result = mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/model",
    name=model_name
)

print(f"Modelo registrado: {result.name} v{result.version}")
