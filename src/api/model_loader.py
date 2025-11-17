"""
model_loader.py
---------------
Carga el modelo entrenado de Spark (GBTR) desde Google Cloud Storage.
"""

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


MODEL_PATH = "gs://nyc-taxi-etl/models/nyc_taxi_fare_gbt_2015_01"


def load_spark_model():
    """
    Carga el modelo entrenado desde GCS.
    Se ejecuta una vez cuando se levanta la API.
    """
    spark = (
        SparkSession.builder
        .appName("NYC Taxi API - Load Model")
        .getOrCreate()
    )

    model = PipelineModel.load(MODEL_PATH)

    print("Modelo cargado correctamente desde:", MODEL_PATH)

    return spark, model
