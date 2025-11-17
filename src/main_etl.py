"""
main_etl.py
===========

Pipeline completo para el dataset NYC Yellow Taxi (enero 2015):

1. Lee datos crudos desde Google Cloud Storage (GCS).
2. Aplica limpieza y transformaciones.
3. Optimiza el procesamiento (particionamiento, cache, broadcast join).
4. Genera datos curados y agregados y los guarda en GCS en formato Parquet.
5. Entrena un modelo de ML (regresión) para predecir el total de la tarifa.
6. Guarda el modelo entrenado en GCS.

Este script está pensado para ejecutarse en un clúster Dataproc:

    python3 src/main_etl.py
"""

import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import broadcast

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator


# ==========================
# 0. RUTAS EN GOOGLE STORAGE
# ==========================

# Ruta de datos crudos
GCS_RAW_PATH = "gs://nyc-taxi-etl/raw/nyc_taxi/yellow_tripdata_2015-01.csv"

# Carpeta para datos curados
GCS_CURATED_PATH = "gs://nyc-taxi-etl/curated/nyc_taxi/yellow_2015_01"

# Carpeta para datos agregados
GCS_AGG_TRIPS_BY_HOUR = "gs://nyc-taxi-etl/agg/nyc_taxi/trips_by_hour_2015_01"

# Carpeta para guardar el modelo de ML entrenado
GCS_MODEL_PATH = "gs://nyc-taxi-etl/models/nyc_taxi_fare_gbt_2015_01"


# ==========================
# 1. CREAR SPARK SESSION
# ==========================

def build_spark_session() -> SparkSession:
    """
    Crea la sesión de Spark.

    En Dataproc ya viene configurado el conector de GCS, por lo que
    no es necesario agregar jars manualmente.
    """
    spark = (
        SparkSession.builder
        .appName("NYC Taxi ETL + ML")
        .getOrCreate()
    )

    # Nivel de log más amigable
    spark.sparkContext.setLogLevel("WARN")

    return spark


# ==========================
# 2. LECTURA DE DATOS CRUDOS
# ==========================

def read_raw_data(spark: SparkSession):
    """
    Lee el CSV crudo desde GCS.

    Usamos inferSchema para que Spark detecte tipos automáticamente.    
    """
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


# ==================================
# 3. LIMPIEZA Y TRANSFORMACIONES
# ==================================

def clean_and_transform(df_raw):
    """
    Aplica reglas de limpieza y crea nuevas columnas.

    - Filtra viajes con valores imposibles o extremos.
    - Calcula duración del viaje, velocidad promedio.
    - Extrae fecha, hora y día de la semana del pickup.
    """

    # Asegurarnos de que los campos de fecha/hora sean timestamp
    df = (
        df_raw
        .withColumn(
            "tpep_pickup_datetime",
            F.col("tpep_pickup_datetime").cast("timestamp")
        )
        .withColumn(
            "tpep_dropoff_datetime",
            F.col("tpep_dropoff_datetime").cast("timestamp")
        )
    )

    # Cálculo de duración del viaje en minutos
    df = df.withColumn(
        "trip_duration_min",
        (F.col("tpep_dropoff_datetime").cast("long") -
         F.col("tpep_pickup_datetime").cast("long")) / 60.0
    )

    # Reglas de filtrado básicas
    df = df.filter(F.col("trip_distance") > 0)
    df = df.filter(F.col("fare_amount") > 0)
    df = df.filter(F.col("total_amount") > 0)
    df = df.filter(F.col("passenger_count") > 0)

    # Duraciones razonables: entre 1 minuto y 3 horas
    df = df.filter((F.col("trip_duration_min") >= 1) &
                   (F.col("trip_duration_min") <= 180))

    # Coordenadas razonables para NYC (aproximadas)
    df = df.filter(
        (F.col("pickup_longitude") > -75) &
        (F.col("pickup_longitude") < -72) &
        (F.col("dropoff_longitude") > -75) &
        (F.col("dropoff_longitude") < -72) &
        (F.col("pickup_latitude") > 40) &
        (F.col("pickup_latitude") < 42) &
        (F.col("dropoff_latitude") > 40) &
        (F.col("dropoff_latitude") < 42)
    )

    # Columnas de fecha / tiempo
    df = df.withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
    df = df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    df = df.withColumn("pickup_dow", F.date_format("tpep_pickup_datetime", "E"))

    # Velocidad promedio (km/h)
    df = df.withColumn(
        "avg_speed_kmh",
        F.col("trip_distance") / (F.col("trip_duration_min") / 60.0)
    )

    # Filtrar velocidades "humanas" (0–120 km/h)
    df = df.filter((F.col("avg_speed_kmh") > 0) &
                   (F.col("avg_speed_kmh") < 120))

    # Pequeña tabla de look-up para payment_type (broadcast join)
    payment_lookup = df.sparkSession.createDataFrame(
        [
            (1, "Credit card"),
            (2, "Cash"),
            (3, "No charge"),
            (4, "Dispute"),
            (5, "Unknown"),
            (6, "Voided trip"),
        ],
        schema=T.StructType([
            T.StructField("payment_type", T.IntegerType(), False),
            T.StructField("payment_desc", T.StringType(), True),
        ])
    )

    # Broadcast join porque la tabla de códigos es muy pequeña
    df = (
        df.join(
            broadcast(payment_lookup),
            on="payment_type",
            how="left"
        )
    )

    print("Datos limpios, ejemplo:")
    df.show(5, truncate=False)

    print("Esquema después de limpieza / transformaciones:")
    df.printSchema()

    return df


# ==================================
# 4. ESCRITURA DE DATOS CURADOS Y AGREGADOS
# ==================================

def write_curated_and_aggregated(df_clean):
    """
    Escribe:
      - Datos curados en Parquet, particionados por fecha.
      - Agregados de viajes por hora.
    """

    # Cacheamos porque lo usaremos para varias operaciones
    df_clean = df_clean.cache()

    print("Número de registros después de limpieza:")
    print(df_clean.count())

    # --- 4.1. Datos curados (granularidad viaje) ---

    print(f"Escribiendo datos curados en: {GCS_CURATED_PATH}")

    (
        df_clean
        .repartition("pickup_date")  # optimización: particionar por fecha
        .write
        .mode("overwrite")
        .partitionBy("pickup_date")
        .parquet(GCS_CURATED_PATH)
    )

    # --- 4.2. Agregado: viajes por hora, distancia y tarifa promedio ---

    print("Generando agregados de viajes por hora...")

    trips_by_hour = (
        df_clean
        .groupBy("pickup_date", "pickup_hour")
        .agg(
            F.count("*").alias("total_trips"),
            F.avg("trip_distance").alias("avg_distance_mi"),
            F.avg("total_amount").alias("avg_total_amount"),
            F.avg("trip_duration_min").alias("avg_duration_min")
        )
    )

    print("Ejemplo de trips_by_hour:")
    trips_by_hour.show(10)

    print(f"Escribiendo agregados en: {GCS_AGG_TRIPS_BY_HOUR}")

    (
        trips_by_hour
        .repartition("pickup_date")
        .write
        .mode("overwrite")
        .partitionBy("pickup_date")
        .parquet(GCS_AGG_TRIPS_BY_HOUR)
    )


# ==================================
# 5. MODELO DE ML: PREDICCIÓN DE TOTAL_AMOUNT
# ==================================

def train_fare_model(df_clean):
    """
    Entrena un modelo de regresión para predecir `total_amount`.

    Tipo de modelo: Gradient Boosted Trees (GBTRegressor).
    Tipo de problema: **Regresión**.

    Features usadas:
      - trip_distance
      - trip_duration_min
      - passenger_count
      - pickup_hour
      - payment_type (one-hot)
    """

    print("Preparando datos para el modelo de ML...")

    # Seleccionamos solo las columnas necesarias y eliminamos nulos
    ml_df = (
        df_clean
        .select(
            "trip_distance",
            "trip_duration_min",
            "passenger_count",
            "pickup_hour",
            "payment_type",
            "total_amount"
        )
        .dropna()
    )

    # Filtro extra para evitar outliers enormes en la etiqueta
    ml_df = ml_df.filter(ml_df.total_amount < 200)

    # Para acelerar el ejemplo podemos tomar una muestra aleatoria
    ml_df = ml_df.sample(withReplacement=False, fraction=0.3, seed=42)

    print("Registros usados para ML:", ml_df.count())

    # Columnas categóricas y numéricas
    categorical_cols = ["payment_type"]
    numeric_cols = ["trip_distance", "trip_duration_min",
                    "passenger_count", "pickup_hour"]

    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_idx",
            handleInvalid="keep"
        )
        for col in categorical_cols
    ]

    encoders = [
        OneHotEncoder(
            inputCols=[f"{col}_idx"],
            outputCols=[f"{col}_ohe"]
        )
        for col in categorical_cols
    ]

    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{c}_ohe" for c in categorical_cols],
        outputCol="features"
    )

    gbt = GBTRegressor(
        labelCol="total_amount",
        featuresCol="features",
        maxIter=20,
        maxDepth=5
    )

    pipeline = Pipeline(
        stages=indexers + encoders + [assembler, gbt]
    )

    train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)

    print("Entrenando modelo GBTRegressor...")

    t0 = time.perf_counter()
    model = pipeline.fit(train_df)
    train_time = time.perf_counter() - t0

    print(f"Tiempo de entrenamiento: {train_time:.2f} segundos")

    # Evaluación en test
    predictions = model.transform(test_df)

    evaluator = RegressionEvaluator(
        labelCol="total_amount",
        predictionCol="prediction",
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(
        labelCol="total_amount",
        predictionCol="prediction",
        metricName="mae"
    ).evaluate(predictions)

    print(f"Métricas del modelo en test:")
    print(f"    RMSE = {rmse:.2f}")
    print(f"    MAE  = {mae:.2f}")

    # Importancias de features (último stage es el GBTRegressor)
    gbt_model = model.stages[-1]
    print("Importancia de features (orden en el vector):")
    print(gbt_model.featureImportances)

    # Guardar modelo en GCS
    print(f"Guardando modelo en: {GCS_MODEL_PATH}")
    model.write().overwrite().save(GCS_MODEL_PATH)

    # Devuelve métricas
    return {
        "train_time_sec": train_time,
        "rmse": rmse,
        "mae": mae,
    }


# ==========================
# 6. FUNCIÓN MAIN
# ==========================

def main():
    spark = build_spark_session()

    overall_start = time.perf_counter()

    # Paso 1: lectura
    t0 = time.perf_counter()
    df_raw = read_raw_data(spark)
    read_time = time.perf_counter() - t0

    # Paso 2: limpieza
    t0 = time.perf_counter()
    df_clean = clean_and_transform(df_raw)
    clean_time = time.perf_counter() - t0

    # Paso 3: escritura de datos curados + agregados
    t0 = time.perf_counter()
    write_curated_and_aggregated(df_clean)
    write_time = time.perf_counter() - t0

    # Paso 4: modelo de ML
    ml_metrics = train_fare_model(df_clean)

    overall_time = time.perf_counter() - overall_start

    print("=" * 80)
    print("RESUMEN DE TIEMPOS")
    print(f"  Lectura CSV           : {read_time:.2f} s")
    print(f"  Limpieza/transform    : {clean_time:.2f} s")
    print(f"  Escritura Parquet     : {write_time:.2f} s")
    print(f"  Entrenamiento modelo  : {ml_metrics['train_time_sec']:.2f} s")
    print(f"  Tiempo total pipeline : {overall_time:.2f} s")
    print("=" * 80)

    spark.stop()


if __name__ == "__main__":
    main()
