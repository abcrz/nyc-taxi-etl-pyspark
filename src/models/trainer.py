# src/models/trainer.py

from pyspark.sql import DataFrame, functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator


def train_fare_model(df: DataFrame, model_path: str) -> dict:
    """
    Entrena un modelo de predicción de tarifa (GBTRegressor) a partir de la capa curated.

    IMPORTANTE:
    - Este método asume que 'df' viene de la capa curated (ya limpia).
    - Para no matar la VM, entrena sobre un subconjunto (sample + limit).
    """

    # ------------------------------------------------------------------
    # 1) Seleccionar solo las columnas necesarias
    # ------------------------------------------------------------------
    cols_needed = [
        "trip_distance",
        "trip_duration_min",
        "passenger_count",
        "pickup_hour",
        "payment_type",
        "total_amount",
    ]
    ml_df = df.select(*cols_needed)

    # ------------------------------------------------------------------
    # 2) Filtros básicos de calidad / outliers
    # ------------------------------------------------------------------
    ml_df = ml_df.filter(
        (F.col("trip_distance") > 0) & (F.col("trip_distance") < 100) &
        (F.col("trip_duration_min") > 0) & (F.col("trip_duration_min") < 240) &
        (F.col("total_amount") > 0) & (F.col("total_amount") < 200)
    )
    ml_df = ml_df.dropna()

    total_ml = ml_df.count()
    print(f"[ML] Registros disponibles antes de sample: {total_ml}")

    # ------------------------------------------------------------------
    # 3) SUBCONJUNTO para no matar la máquina
    #    - Sample 2% de la data
    #    - Máximo 300k filas
    # ------------------------------------------------------------------
    ml_df = ml_df.sample(withReplacement=False, fraction=0.02, seed=42)
    ml_df = ml_df.limit(300_000)

    sampled_count = ml_df.count()
    print(f"[ML] Registros después de sample/limit: {sampled_count}")

    if sampled_count == 0:
        raise ValueError("No hay registros suficientes para entrenar el modelo.")

    # ------------------------------------------------------------------
    # 4) Train / test split
    # ------------------------------------------------------------------
    train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
    print(f"[ML] Registros train: {train_df.count()}  |  test: {test_df.count()}")

    # ------------------------------------------------------------------
    # 5) Pipeline de features + modelo
    # ------------------------------------------------------------------
    # Categórica: payment_type
    indexer = StringIndexer(
        inputCol="payment_type",
        outputCol="payment_type_index",
        handleInvalid="keep",
    )

    encoder = OneHotEncoder(
        inputCols=["payment_type_index"],
        outputCols=["payment_type_ohe"],
    )

    assembler = VectorAssembler(
        inputCols=[
            "trip_distance",
            "trip_duration_min",
            "passenger_count",
            "pickup_hour",
            "payment_type_ohe",
        ],
        outputCol="features",
    )

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="total_amount",
        maxDepth=5,
        maxIter=60,
        stepSize=0.1,
    )

    pipeline = Pipeline(stages=[indexer, encoder, assembler, gbt])

    # ------------------------------------------------------------------
    # 6) Entrenamiento
    # ------------------------------------------------------------------
    print("[ML] Entrenando modelo GBT...")
    pipeline_model = pipeline.fit(train_df)
    print("[ML] Modelo entrenado.")

    # ------------------------------------------------------------------
    # 7) Evaluación
    # ------------------------------------------------------------------
    preds = pipeline_model.transform(test_df)

    evaluator_rmse = RegressionEvaluator(
        labelCol="total_amount",
        predictionCol="prediction",
        metricName="rmse",
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="total_amount",
        predictionCol="prediction",
        metricName="mae",
    )

    rmse = evaluator_rmse.evaluate(preds)
    mae = evaluator_mae.evaluate(preds)

    print(f"[ML] RMSE: {rmse:.3f}  |  MAE: {mae:.3f}")

    # ------------------------------------------------------------------
    # 8) Guardar modelo en GCS
    # ------------------------------------------------------------------
    print(f"[ML] Guardando modelo en: {model_path}")
    (
        pipeline_model
        .write()
        .overwrite()
        .save(model_path)
    )

    return {"rmse": rmse, "mae": mae}
