# Databricks notebook
from pyspark.sql import SparkSession
from databricks.feature_store import FeatureStoreClient

spark = SparkSession.builder.getOrCreate()
fs = FeatureStoreClient()

# Datos de ejemplo
data = [
    {"id": 1, "feature_1": 10, "feature_2": 100, "label": 1},
    {"id": 2, "feature_1": 20, "feature_2": 200, "label": 0},
    {"id": 3, "feature_1": 30, "feature_2": 300, "label": 1},
]

df = spark.createDataFrame(data)

# Crear tabla de features
feature_table_name = "ml_catalog.ml_schema.ft_example_model"

fs.create_table(
    name=feature_table_name,
    primary_keys=["id"],
    schema=df.schema,
    description="Tabla de features para modelo de prueba",
    force_destroy=True
)

fs.write_table(
    name=feature_table_name,
    df=df,
    mode="overwrite"
)

print(f"Tabla de features creada: {feature_table_name}")
