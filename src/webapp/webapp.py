"""
webapp.py
---------
Interfaz web (HTML) para el modelo NYC Taxi.

- Renderiza src/webapp/templates/index.html
- Recibe los datos del formulario
- Calcula una duración estimada del viaje (trip_duration_min)
- Usa el modelo de Spark cargado por models.model_loader
"""

from flask import Flask, render_template, request
from pyspark.sql import Row

from ..models.model_loader import load_spark_model

app = Flask(__name__)

# Cargamos Spark + modelo UNA sola vez al iniciar la webapp
spark, model = load_spark_model()

# Velocidad promedio aproximada de un yellow cab en NYC (millas/hora)
AVG_SPEED_MPH = 12.0  # puedes ajustar este valor si quieres


@app.route("/", methods=["GET"])
def home():
    # Muestra el formulario vacío
    return render_template("index.html")


@app.route("/predict_web", methods=["POST"])
def predict_web():
    """
    Toma los datos del formulario HTML,
    calcula una duración estimada y genera la predicción.
    """
    try:
        # 1. Leer datos del formulario
        trip_distance = float(request.form["trip_distance"])
        passenger_count = int(request.form["passenger_count"])
        pickup_hour = int(request.form["pickup_hour"])
        payment_type = int(request.form["payment_type"])

        # 2. Calcular duración estimada (min) a partir de la distancia
        #    duración (horas) = distancia / velocidad
        #    duración (min)   = horas * 60
        if AVG_SPEED_MPH > 0:
            duration_est = (trip_distance / AVG_SPEED_MPH) * 60.0
        else:
            duration_est = 10.0  # valor fallback raro, pero evita división por cero

        trip_duration_min = duration_est

        # 3. Row → DataFrame de Spark con las mismas features que el modelo espera
        input_row = Row(
            trip_distance=trip_distance,
            trip_duration_min=trip_duration_min,
            passenger_count=passenger_count,
            pickup_hour=pickup_hour,
            payment_type=payment_type,
        )

        df = spark.createDataFrame([input_row])

        # 4. Obtener la predicción del modelo
        pred_row = model.transform(df).select("prediction").first()
        prediction = round(float(pred_row["prediction"]), 2)

        # 5. Renderizar la página con resultados
        return render_template(
            "index.html",
            prediction=prediction,
            trip_distance=trip_distance,
            passenger_count=passenger_count,
            pickup_hour=pickup_hour,
            duration_est=round(duration_est, 1),
        )

    except Exception as e:
        print("Error en /predict_web:", e)
        return render_template("index.html", error=str(e)), 500


if __name__ == "__main__":
    # sin debug=True para no duplicar SparkSession
    app.run(host="0.0.0.0", port=5000)
