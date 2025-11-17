"""
webapp.py
=========

Aplicación Flask que expone una interfaz web para consumir el modelo de
regresión entrenado en PySpark (GBTRegressor) para predecir la tarifa
total de un viaje de taxi en NYC.

Arquitectura general:
- PySpark se usa para cargar el PipelineModel previamente entrenado.
- Flask expone una ruta web con un formulario.
- El usuario ingresa distancia, pasajeros, hora y tipo de pago.
- El backend estima la duración del viaje y construye un DataFrame de Spark.
- El modelo genera la predicción y se devuelve a la plantilla HTML.
"""

from flask import Flask, render_template, request

from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel


# ============================================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================================

# Ruta en Google Cloud Storage donde está guardado el modelo entrenado
MODEL_PATH = "gs://nyc-taxi-etl/models/nyc_taxi_fare_gbt_2015_01"

# Instancia principal de la aplicación Flask
app = Flask(__name__)


# ============================================================================
# 2. SESIÓN DE SPARK COMPARTIDA
# ============================================================================

def build_spark_session() -> SparkSession:
    """
    Crea o recupera una SparkSession compartida para toda la aplicación.

    En Dataproc ya viene configurado el conector de GCS, por lo que
    no es necesario añadir jars adicionales.
    """
    spark = (
        SparkSession.builder
        .appName("NYC Taxi WebApp")
        .getOrCreate()
    )
    # Nivel de log menos ruidoso
    spark.sparkContext.setLogLevel("WARN")
    return spark


# Creamos la sesión de Spark una sola vez al iniciar el módulo
spark: SparkSession = build_spark_session()


# ============================================================================
# 3. CARGA DEL MODELO DE ML
# ============================================================================

def load_model(path: str) -> PipelineModel:
    """
    Carga el modelo de ML (PipelineModel) desde la ruta indicada.

    Parámetros
    ----------
    path : str
        Ruta en GCS donde está guardado el modelo.

    Retorna
    -------
    PipelineModel
        Modelo listo para usarse en transformaciones.
    """
    print("Cargando modelo desde GCS…")
    model = PipelineModel.load(path)
    print("Modelo cargado correctamente.")
    return model


model: PipelineModel = load_model(MODEL_PATH)


# ============================================================================
# 4. FUNCIÓN AUXILIAR: ESTIMACIÓN DE DURACIÓN
# ============================================================================

def estimate_duration(distance_miles: float) -> float:
    """
    Estima la duración del viaje en minutos a partir de la distancia.

    Suposición:
    - Velocidad promedio de un taxi en NYC ~ 12 mph.

    Parámetros
    ----------
    distance_miles : float
        Distancia del viaje en millas.

    Retorna
    -------
    float
        Duración estimada en minutos (redondeada a un decimal).
    """
    avg_speed_mph = 12.0
    duration_min = (distance_miles / avg_speed_mph) * 60.0
    return round(duration_min, 1)


# ============================================================================
# 5. RUTAS DE FLASK
# ============================================================================

@app.route("/", methods=["GET"])
def home():
    """
    Renderiza la página principal con el formulario.

    No se envía ninguna predicción en este punto; la plantilla se
    encarga de mostrar solo el formulario cuando no hay resultados.
    """
    return render_template("index.html")


@app.route("/predict_web", methods=["POST"])
def predict_web():
    """
    Recibe los datos del formulario web, construye un DataFrame de Spark
    con una sola fila y aplica el modelo de ML para obtener la tarifa
    estimada.

    Flujo:
    1. Leer valores enviados desde el formulario HTML.
    2. Calcular la duración estimada del viaje a partir de la distancia.
    3. Construir un Row y convertirlo en DataFrame de Spark.
    4. Ejecutar model.transform(df) para obtener la predicción.
    5. Renderizar nuevamente la plantilla con:
       - la predicción
       - los valores ingresados
       - la duración estimada
    """
    try:
        # ------------------------------------------------------------
        # 1. Lectura y conversión de datos del formulario
        # ------------------------------------------------------------
        distance = float(request.form["trip_distance"])
        passengers = int(request.form["passenger_count"])
        hour = int(request.form["pickup_hour"])
        payment = int(request.form["payment_type"])

        # Duración estimada del viaje (en minutos) usando la función auxiliar
        duration_est = estimate_duration(distance)

        # ------------------------------------------------------------
        # 2. Construcción del diccionario de valores de entrada
        #    Debe ser consistente con las columnas que espera el modelo.
        # ------------------------------------------------------------
        vals = {
            "trip_distance": distance,
            "trip_duration_min": duration_est,
            "passenger_count": passengers,
            "pickup_hour": hour,
            "payment_type": payment,
        }

        # ------------------------------------------------------------
        # 3. DataFrame de una sola fila para alimentar el modelo
        # ------------------------------------------------------------
        df = spark.createDataFrame([Row(**vals)])

        # ------------------------------------------------------------
        # 4. Transformación y obtención de predicción
        # ------------------------------------------------------------
        pred = model.transform(df).collect()[0]["prediction"]
        pred_redondeada = round(float(pred), 2)

        # ------------------------------------------------------------
        # 5. Renderizar plantilla con todos los datos necesarios
        # ------------------------------------------------------------
        return render_template(
            "index.html",
            prediction=pred_redondeada,
            trip_distance=distance,
            passenger_count=passengers,
            pickup_hour=hour,
            duration_est=duration_est,
        )

    except Exception as exc:
        # En un entorno real se debería registrar el error (logging)
        # y devolver una página de error más amigable.
        return f"Error interno en la predicción: {exc}"


# ============================================================================
# 6. PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # host="0.0.0.0" permite que la aplicación sea accesible desde fuera
    # de la VM (por ejemplo, usando port forwarding desde tu equipo local).
    app.run(host="0.0.0.0", port=8081)
