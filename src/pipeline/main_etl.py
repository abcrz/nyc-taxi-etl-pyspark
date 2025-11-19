"""
Orquestador del ETL NYC Taxi:

1. Lee datos crudos desde GCS.
2. Aplica limpieza y transformaciones.
3. Escribe capas curated y agregada en GCS.

Este script NO entrena el modelo. El entrenamiento
se ejecuta en un script separado (main_train.py).
"""

import os, sys
import time
from pyspark.sql import SparkSession

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
from gcs.paths import (
    GCS_RAW_PATH,
    GCS_CURATED_PATH,
    GCS_AGG_TRIPS_BY_HOUR,
)
from features.transformations import clean_and_transform
from pipeline.etl_writer import write_curated, write_aggregates


def read_raw_data(spark: SparkSession):
    """Lee el CSV crudo desde GCS con encabezados y schema inferido."""
    print(f"Leyendo datos crudos desde: {GCS_RAW_PATH}")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(GCS_RAW_PATH)
    )

    print("Ejemplo de datos crudos:")
    df.show(5, truncate=False)

    print("Esquema crudo:")
    df.printSchema()

    return df


def main():
    # 0. Crear sesión de Spark
    spark = build_spark_session("NYC Taxi ETL (script)")

    overall_start = time.perf_counter()

    # 1. Lectura
    t0 = time.perf_counter()
    df_raw = read_raw_data(spark)
    read_time = time.perf_counter() - t0

    # 2. Limpieza / transformaciones
    t0 = time.perf_counter()
    df_clean = clean_and_transform(df_raw)
    clean_time = time.perf_counter() - t0

    # 3. Escritura de datos curated y agregados
    t0 = time.perf_counter()
    print("Número de registros después de limpieza:", df_clean.count())

    # 3.1 curated con toda la data
    write_curated(df_clean, GCS_CURATED_PATH)

    # 3.2 agregados solo con 5% de la data para no matar la VM
    write_aggregates(df_clean, GCS_AGG_TRIPS_BY_HOUR, sample_fraction=0.05)

    write_time = time.perf_counter() - t0

    overall_time = time.perf_counter() - overall_start

    print("=" * 80)
    print("RESUMEN DE TIEMPOS ETL")
    print(f"  Lectura CSV        : {read_time:.2f} s")
    print(f"  Limpieza/transform : {clean_time:.2f} s")
    print(f"  Escritura Parquet  : {write_time:.2f} s")
    print(f"  Tiempo total ETL   : {overall_time:.2f} s")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
