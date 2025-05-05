# Databricks notebook
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from databricks.feature_store import FeatureStoreClient, FeatureLookup

fs = FeatureStoreClient()

feature_table = "ml_catalog.ml_schema.ft_example_model"

feature_lookups = [FeatureLookup(table_name=feature_table, lookup_key="id")]

training_set = fs.create_training_set(
    df=fs.read_table(feature_table),
    feature_lookups=feature_lookups,
    label="label"
)

df = training_set.load_df().toPandas()
X = df.drop(columns=["label", "id"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

experiment_path = os.getenv("DATABRICKS_EXPERIMENT_PATH", "/Shared/default_experiment")
model_name = os.getenv("DATABRICKS_MODEL_NAME", "default_model_name")

mlflow.set_experiment(experiment_path)

with mlflow.start_run(run_name="train_model"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.log_metric("accuracy", acc)
