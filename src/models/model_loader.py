"""
model_loader.py
---------------
Carga el modelo entrenado de Spark (GBTR) desde Google Cloud Storage.
Se utiliza tanto por la API (src/api/app.py) como por la webapp
(src/webapp/webapp.py).
"""

import os
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

# Import relativo dentro del paquete src
from ..gcs.paths import GCS_MODEL_PATH


def load_spark_model():
    """
    Carga el modelo entrenado desde GCS y devuelve (spark, model).

    - Fuerza Spark a ejecutarse en modo local[*] (sin YARN).
    - Debe llamarse solo una vez por proceso (API o webapp).
    """

    # Forzar ejecución local de Spark (evita que intente usar YARN)
    os.environ["SPARK_MASTER"] = "local[*]"
    os.environ["PYSPARK_SUBMIT_ARGS"] = "--master local[*] pyspark-shell"

    spark = (
        SparkSession.builder
        .appName("NYC Taxi - Load Trained Model")
        .master("local[*]")          # aseguramos modo local
        .getOrCreate()
    )

    print("[MODEL_LOADER] Cargando modelo desde:", GCS_MODEL_PATH)

    model = PipelineModel.load(GCS_MODEL_PATH)

    print("[MODEL_LOADER] Modelo cargado correctamente.")

    return spark, model
