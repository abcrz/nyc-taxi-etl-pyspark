# src/utils/spark_builder.py
from pyspark.sql import SparkSession

def build_spark_session(app_name="NYC Taxi", master="local[*]"):
    """
    Crea la sesión de Spark.

    - master="local[*]"  -> usa todos los cores de la máquina local.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)   
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
