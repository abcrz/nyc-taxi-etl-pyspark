"""
app.py
------
API REST en Flask para exponer el modelo de ML entrenado en PySpark.

Endpoint:
    POST /predict

Ejemplo de entrada:
{
    "trip_distance": 3.2,
    "trip_duration_min": 14.5,
    "passenger_count": 1,
    "pickup_hour": 18,
    "payment_type": 1
}
"""

from flask import Flask, request, jsonify
from pyspark.sql import Row
from pyspark.sql import SparkSession
from ..models.model_loader import load_spark_model

# Cargamos modelo de Spark ML al iniciar la API
spark, model = load_spark_model()

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Realiza predicción de tarifa usando el modelo entrenado.
    """

    try:
        data = request.get_json()

        required_fields = [
            "trip_distance",
            "trip_duration_min",
            "passenger_count",
            "pickup_hour",
            "payment_type"
        ]

        for f in required_fields:
            if f not in data:
                return jsonify({"error": f"Missing field: {f}"}), 400

        # Convertimos dict → Row (Spark)
        input_row = Row(
            trip_distance=float(data["trip_distance"]),
            trip_duration_min=float(data["trip_duration_min"]),
            passenger_count=int(data["passenger_count"]),
            pickup_hour=int(data["pickup_hour"]),
            payment_type=int(data["payment_type"])
        )

        df = spark.createDataFrame([input_row])

        # Generar predicción
        pred = model.transform(df).collect()[0]["prediction"]

        return jsonify({
            "prediction_total_amount": round(float(pred), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return {"status": "API NYC Taxi ML Model OK"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
