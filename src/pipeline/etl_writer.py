# src/pipeline/etl_writer.py
from pyspark.sql import functions as F

def write_curated(df, path):
    print(f"Escribiendo datos curated en: {path}")
    (
        df.repartition("pickup_date")  # particionamos por fecha
          .write
          .mode("overwrite")
          .partitionBy("pickup_date")
          .parquet(path)
    )


def write_aggregates(df, path, sample_fraction=0.05):
    """
    Escribe agregados de viajes por hora.
    - Por defecto usa solo un 5% de la data para no matar la VM.
    """
    if sample_fraction < 1.0:
        print(f"Generando agregados a partir de una muestra del {sample_fraction*100:.1f}% de los datos...")
        df = df.sample(withReplacement=False, fraction=sample_fraction, seed=42)

    trips_by_hour = (
        df.groupBy("pickup_date", "pickup_hour")
          .agg(
              F.count("*").alias("total_trips"),
              F.avg("trip_distance").alias("avg_distance_mi"),
              F.avg("total_amount").alias("avg_total_amount"),
              F.avg("trip_duration_min").alias("avg_duration_min"),
          )
    )

    print("Ejemplo de trips_by_hour:")
    trips_by_hour.orderBy("pickup_date", "pickup_hour").show(10)

    print(f"Escribiendo agregados en: {path}")
    (
        trips_by_hour
          .coalesce(4)     # reducimos número de archivos / shuffles
          .write
          .mode("overwrite")
          .partitionBy("pickup_date")
          .parquet(path)
    )
