# NYC Taxi ETL con PySpark, Google Cloud Dataproc y WebApp de Predicción

Proyecto de analítica y machine learning para el dataset de taxis de Nueva York.  
Incluye un **pipeline ETL en PySpark sobre Dataproc**, almacenamiento en **Google Cloud Storage (GCS)**, un **módulo de entrenamiento de modelo** y una **API + web app Flask** para estimar tarifas de viaje en tiempo casi real.

> Bootcamp Python para Ciencia de Datos – Módulo 4: Big Data, MLOps

---

## Objetivo del proyecto

El objetivo es construir de punta a punta un flujo típico de **Data Engineering + ML**:

1. Ingerir datos del dataset público de NYC Taxi.
2. Procesarlos con PySpark en un **cluster Dataproc** y/o Spark local.
3. Generar capas **raw → curated → agg** en un bucket de **GCS**.
4. Entrenar un modelo de predicción de tarifa (Gradient Boosted Trees).
5. Publicar el modelo como servicio mediante:
   - Una **API REST** (`/predict`) para consumo programático.
   - Una **web app Flask** con interfaz HTML interactiva para el usuario final.

---

## Problema que se aborda

Las ciudades generan millones de registros de viajes de taxi.  
Sin una buena analítica:

- Es difícil **estimar tarifas** antes del viaje.
- No se aprovechan patrones espaciales/temporales (zona, hora, distancia, horario).
- Se pierden oportunidades de **optimizar demanda, rutas y costos**.

---

## Solución propuesta

Este proyecto implementa:

- Un **pipeline ETL en PySpark** que limpia, filtra y transforma los datos históricos.
- Un **modelo de ML** que predice la tarifa del viaje a partir de features como:
  - Distancia del trayecto  
  - Duración estimada del viaje  
  - Hora del viaje  
  - Cantidad de pasajeros  
  - Tipo de pago
- Un **bucket de GCS** que organiza todo el ciclo de vida del dato:
  - `raw/`, `curated/`, `agg/`, `models/`.
- Una **API REST en Flask** que:
  - Expone un endpoint JSON `/predict` para obtener la tarifa estimada.
- Una **web app Flask** que:
  - Muestra un formulario para capturar los datos del viaje.
  - Calcula una duración estimada según la distancia.
  - Consulta el modelo entrenado (cargado desde GCS).
  - Devuelve una **tarifa estimada** y un resumen visual (chips, gauge, mapa).

---

## Arquitectura general

```text
           Dataset público NYC Taxi
                     │
                     ▼
             Google Cloud Storage (GCS)
             ├─ raw/
             ├─ curated/
             ├─ agg/
             └─ models/
                     ▲
                     │
        PySpark ETL en Dataproc / Spark local
                     │
       ┌─────────────┴─────────────┐
       │                           │
  Limpieza + FE               Entrenamiento modelo
(main_etl.py / etl.ipynb)    (main_train.py / train_model.ipynb)
                                   │
                                   ▼
                     Modelo GBT guardado en GCS
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
              API REST Flask               Web App Flask
               (src/api/app.py)          (src/webapp/webapp.py)
                     │                           │
                     ▼                           ▼
         /predict (JSON)               UI HTML (templates/index.html)
         Integraciones                 Formulario + gauge + mapa
```

---

## Infraestructura en GCP

- **Cluster Dataproc:** `nyc-taxi-etl-pyspark` (1 master + workers, imagen 2.2.70-debian12).
- **Bucket GCS:** `nyc-taxi-etl` con la jerarquía:
  - `raw/nyc_taxi/…`
  - `curated/nyc_taxi/…`
  - `agg/nyc_taxi/…`
  - `models/nyc_taxi_fare_gbt_2015_01/…`
- **Red privada + Cloud NAT** para que el cluster tenga salida a internet sin IP pública.
- **Acceso desde local** mediante `gcloud` con tunelización de puertos (port-forwarding) para ver la interfaz web o consumir la API desde el navegador/postman.

Ejemplo de túnel para la webapp (puerto 5000 en la VM → 8081 local):

```bash
gcloud compute ssh nyc-taxi-etl-pyspark-m \
  --zone=us-central1-f \
  --project=advance-wavelet-478419-r5 \
  -- -L 8081:localhost:5000
```

Luego acceder en el navegador local a:

```text
http://localhost:8081
```

---

## Estructura del repositorio

```text
nyc-taxi-etl-pyspark/
│
├─ notebooks/
│   ├─ etl.ipynb              # Exploración y ejecución manual del ETL
│   └─ train_model.ipynb      # Entrenamiento del modelo en modo interactivo
│
├─ src/
│   ├─ pipeline/
│   │   ├─ main_etl.py        # Pipeline ETL: raw → curated → agg
│   │   ├─ main_train.py      # Entrenamiento del modelo y guardado en GCS
│   │   └─ etl_writer.py      # Funciones auxiliares para escritura en GCS
│   │
│   ├─ features/
│   │   └─ transformations.py # Limpieza y feature engineering
│   │
│   ├─ gcs/
│   │   └─ paths.py           # Rutas centralizadas a GCS (raw, curated, agg, models)
│   │
│   ├─ utils/
│   │   └─ spark_builder.py   # Construcción de SparkSession (local/Dataproc)
│   │
│   ├─ models/
│   │   ├─ trainer.py         # Lógica de entrenamiento (GBTRegressor, splits, métricas)
│   │   └─ model_loader.py    # Carga del modelo desde GCS (Spark ML PipelineModel)
│   │
│   ├─ api/
│   │   └─ app.py             # API REST Flask (endpoint /predict)
│   │
│   └─ webapp/
│       ├─ webapp.py          # Web app Flask con formulario HTML
│       └─ templates/
│           └─ index.html     # UI (formulario + chips + gauge + mapa)
│
├─ .gitignore
├─ requirements.txt
└─ README.md                  # Este archivo
```

---

## Tecnologías utilizadas

### Lenguajes y librerías

- **Python 3.x**
- **PySpark** (`pyspark.sql`, `pyspark.ml`)
- **Flask** (API + webapp)
- **Jinja2** (templates HTML)
- **Chart.js** (visualización tipo gauge)
- **Leaflet** (mapa de NYC)
- **Pandas / NumPy** (apoyo ligero en notebooks)

### Google Cloud Platform

- **Dataproc:** ejecución distribuida de PySpark (ETL y entrenamiento).
- **Cloud Storage (GCS):** almacenamiento de datos, capas del modelo y artefactos de Spark ML.
- **VPC + Cloud Router + Cloud NAT:** salida a internet sin exponer directamente nodos del cluster.
- **gcloud CLI:** envío de jobs a Dataproc y tunelización de puertos para desarrollo remoto.

---

## Dataset

Se utilizó el dataset público **Yellow Taxi Trip Records** de la NYC TLC.

Para este proyecto se trabajó con un subconjunto representativo:

- Mes: **enero 2015**
- Formato: CSV (lectura cruda) → Parquet (curated/agg)
- Volumen aproximado: ~12 millones de registros

### Variables más relevantes

- `tpep_pickup_datetime`, `tpep_dropoff_datetime`
- `pickup_longitude`, `pickup_latitude`
- `dropoff_longitude`, `dropoff_latitude`
- `trip_distance`
- `passenger_count`
- `payment_type`
- `fare_amount` (variable objetivo)
- `tolls_amount`, `extra`, `mta_tax`, `improvement_surcharge`, `total_amount`, etc.

Estas variables se procesaron en el pipeline ETL para generar las características que alimentan el modelo de predicción de tarifa.

---

## Pipeline ETL en PySpark (`main_etl.py`)

Resumen de pasos principales:

1. **Configuración del SparkSession**  
   Usando `utils/spark_builder.py`, con las opciones necesarias para leer/escribir en GCS.

2. **Lectura de datos crudos** desde:  
   `gs://nyc-taxi-etl/raw/nyc_taxi/yellow_tripdata_2015-01.csv`

3. **Limpieza y validación** (en `features/transformations.py`):
   - Eliminación de filas con coordenadas inválidas.
   - Filtro de tarifas negativas o ridículamente altas.
   - Filtro de distancias nulas o inconsistentes.
   - Consistencia básica entre tiempos de pick-up y drop-off.

4. **Feature Engineering:**
   - Cálculo de duración del viaje en minutos.
   - Extracción de variables de tiempo: `pickup_hour`, etc.
   - Selección de columnas relevantes y casteos de tipos.

5. **Escritura de la capa curated:**
   - Datos listos para modelado →  
     `gs://nyc-taxi-etl/curated/nyc_taxi/yellow_2015_01`

6. **Agregaciones (agg layer)** usando `etl_writer.py`:
   - Métricas de viajes por hora (`trips_by_hour`), incluyendo:
     - total de viajes
     - distancia promedio
     - monto total promedio
     - duración promedio
   - Escritas en:  
     `gs://nyc-taxi-etl/agg/nyc_taxi/trips_by_hour_2015_01`

---

## Entrenamiento del modelo (`main_train.py` y `models/trainer.py`)

El entrenamiento está desacoplado del ETL y se ejecuta en un script separado:

1. **Lectura de capa curated** desde:  
   `gs://nyc-taxi-etl/curated/nyc_taxi/yellow_2015_01`

2. **Muestreo / limit** (para reducir costo de cómputo en pruebas).

3. **Split train/test** (`randomSplit`) para evaluar desempeño.

4. **Pipeline de ML** (`models/trainer.py`):
   - `VectorAssembler` para features numéricas.
   - Modelo: **`GBTRegressor`** (Gradient Boosted Trees).
   - Entrenamiento en el subconjunto train.
   - Evaluación en test usando:
     - RMSE (Root Mean Squared Error)
     - MAE  (Mean Absolute Error)

5. **Persistencia del modelo** en GCS:

   ```text
   gs://nyc-taxi-etl/models/nyc_taxi_fare_gbt_2015_01/
   ├─ metadata/
   │  └─ stages.json
   └─ stages/
   ```

6. Los scripts imprimen en log las métricas finales, así como tiempos de:
   - lectura,
   - entreno,
   - escritura del modelo.

---

## Carga del modelo y API web

### `models/model_loader.py`

Centraliza la lógica de carga del modelo Spark ML desde GCS:

- Crea un `SparkSession` en modo **local** (para evitar dependencia de YARN cuando se ejecuta en una VM o entorno simple).
- Carga el `PipelineModel` desde `GCS_MODEL_PATH`.
- Devuelve un tuple `(spark, model)` compartido entre API y webapp.

Se utiliza tanto en:

- `src/api/app.py`
- `src/webapp/webapp.py`

### API REST – `src/api/app.py`

- Monta una API Flask con:
  - **GET /** → verificación de estado.
  - **POST `/predict`** → recibe un JSON con:

    ```json
    {
      "trip_distance": 3.2,
      "trip_duration_min": 14.5,
      "passenger_count": 1,
      "pickup_hour": 18,
      "payment_type": 1
    }
    ```

  - Construye un `DataFrame` de Spark a partir del JSON.
  - Aplica el modelo y devuelve un JSON con la tarifa estimada, por ejemplo:

    ```json
    {
      "prediction_total_amount": 12.87
    }
    ```

- Pensado para ser consumido desde Postman, otras apps, o eventualmente desde un frontend separado.

Ejecución local (en la VM):

```bash
python3 -m src.api.app
```

Luego, con túnel:

```bash
gcloud compute ssh nyc-taxi-etl-pyspark-m \
  --zone=us-central1-f \
  --project=advance-wavelet-478419-r5 \
  -- -L 8080:localhost:8080
```

Consumir desde la máquina local:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_distance": 3.2, "trip_duration_min": 14.5, "passenger_count": 1, "pickup_hour": 18, "payment_type": 1}'
```

---

## Web App – `src/webapp/webapp.py` + `templates/index.html`

La webapp ofrece una interfaz amigable para usuarios no técnicos.

### Flujo de la webapp

1. El usuario ingresa:
   - Distancia del viaje (millas)
   - Cantidad de pasajeros
   - Hora del día (0–23)
   - Tipo de pago

2. La webapp calcula internamente una **duración estimada** a partir de la distancia usando una **velocidad promedio** (`AVG_SPEED_MPH`).

3. Se construye un `Row` de Spark con los campos:
   - `trip_distance`
   - `trip_duration_min` (estimada)
   - `passenger_count`
   - `pickup_hour`
   - `payment_type`

4. El modelo predice la tarifa total, y se renderiza nuevamente `index.html` mostrando:

   - Tarifa estimada (`$XX.XX`)
   - Chips con la info del viaje (distancia, pasajeros, hora, duración)
   - Un gauge (Chart.js) que posiciona la tarifa en un rango de referencia.
   - Un mapa (Leaflet) centrado en NYC como contexto visual.

### Ejecución de la webapp

En la VM:

```bash
python3 -m src.webapp.webapp
```

Túnel desde la máquina local:

```bash
gcloud compute ssh nyc-taxi-etl-pyspark-m \
  --zone=us-central1-f \
  --project=advance-wavelet-478419-r5 \
  -- -L 8081:localhost:5000
```

Navegador local:

```text
http://localhost:8081
```

---

## Roadmap / Trabajo futuro

- Integrar la webapp para que consuma directamente la API REST `/predict` (frontend → API → modelo).
- Guardar métricas de entrenamiento en un sistema de tracking (MLflow, BigQuery, etc.).
- Añadir validaciones más robustas en la web app (rangos, mensajes de error UX).
- Desplegar la API y/o la webapp en **Cloud Run** o **GKE** con contenedores Docker.
- Añadir tests unitarios para las funciones clave del ETL y del trainer.
- Incorporar nuevas features (ej. clima, día de la semana, ubicación más granular) para mejorar el modelo.
