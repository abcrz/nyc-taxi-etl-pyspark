from flask import Flask, request, jsonify
from pyspark.sql import Row
from ..models.model_loader import load_spark_model

app = Flask(__name__)

# Cargar Spark + modelo una sola vez
spark, model = load_spark_model()

REQUIRED_FIELDS = {
    "trip_distance": float,
    "trip_duration_min": float,
    "passenger_count": int,
    "pickup_hour": int,
    "payment_type": int
}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validación
        for field, dtype in REQUIRED_FIELDS.items():
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
            try:
                data[field] = dtype(data[field])
            except:
                return jsonify({"error": f"Invalid type for field: {field}"}), 400

        # DataFrame Spark
        df = spark.createDataFrame([Row(**data)])

        # Predicción
        pred = model.transform(df).first().prediction

        return jsonify({
            "prediction_total_amount": round(float(pred), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return {"status": "NYC Taxi API Model Loaded OK"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
