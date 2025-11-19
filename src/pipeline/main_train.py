"""

Script de entrenamiento del modelo NYC Taxi:

1. Lee la capa curated desde GCS.
2. Entrena el modelo de tarifa (GBTRegressor).
3. Guarda el modelo en GCS.

Se asume que el ETL (main_etl.py) ya se ejecutó antes.
"""

import os, sys
import time

# -------------------------------------------------------------------
#  Ajuste del PYTHONPATH
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

print("PROJECT ROOT:", PROJECT_ROOT)
print("SRC PATH    :", SRC_PATH)

# -------------------------------------------------------------------
#  Imports de módulos del proyecto
# -------------------------------------------------------------------
from utils.spark_builder import build_spark_session
from gcs.paths import GCS_CURATED_PATH, GCS_MODEL_PATH
from models.trainer import train_fare_model


def main():
    # 0. Crear sesión de Spark
    spark = build_spark_session("NYC Taxi Training (script)")

    overall_start = time.perf_counter()

    # 1. Lectura de curated
    t0 = time.perf_counter()
    print(f"Leyendo datos curated desde: {GCS_CURATED_PATH}")
    df_for_ml = spark.read.parquet(GCS_CURATED_PATH)
    read_time = time.perf_counter() - t0

    # 2. Entrenamiento del modelo
    t0 = time.perf_counter()
    ml_metrics = train_fare_model(df_for_ml, GCS_MODEL_PATH)
    train_time = time.perf_counter() - t0

    overall_time = time.perf_counter() - overall_start

    print("=" * 80)
    print("RESUMEN ENTRENAMIENTO")
    print(f"  Lectura curated     : {read_time:.2f} s")
    print(f"  Entrenamiento modelo: {train_time:.2f} s")
    print(f"  Tiempo total script : {overall_time:.2f} s")
    print("MÉTRICAS ML:")
    print(f"  RMSE = {ml_metrics['rmse']:.2f}")
    print(f"  MAE  = {ml_metrics['mae']:.2f}")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
