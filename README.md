# NYC Taxi ETL con PySpark & Google Cloud Dataproc

Proyecto de analítica y machine learning para el dataset de taxis de Nueva York.  
Incluye un **pipeline ETL en PySpark sobre Dataproc**, almacenamiento en **Google Cloud Storage (GCS)** y una **web app Flask** para estimar tarifas de viaje en tiempo casi real.

> Bootcamp Python para Ciencia de Datos – Módulo 4: Big Data, MLOps

---

## Objetivo del proyecto

El objetivo es construir de punta a punta un flujo típico de **Data Engineering + ML**:

1. Ingerir datos del dataset público de NYC Taxi.
2. Procesarlos con PySpark en un **cluster Dataproc**.
3. Generar capas **raw → curated → agg** en un bucket de **GCS**.
4. Entrenar un modelo de predicción de tarifa (Gradient Boosted Trees).
5. Publicar el modelo como servicio con una **API web** y una interfaz sencilla para el usuario.

---

## Problema que se aborda

Las ciudades generan millones de registros de viajes de taxi.  
Sin una buena analítica:

- Es difícil **estimar tarifas** antes del viaje.
- No se aprovechan patrones espaciales/temporales (zona, hora, distancia).
- Se pierden oportunidades de **optimizar demanda, rutas y costos**.

---

## Solución propuesta

Este proyecto implementa:

- Un **pipeline ETL en PySpark** que limpia, filtra y transforma los datos históricos.
- Un **modelo de ML** que predice la tarifa del viaje a partir de features como:
  - Distancia del trayecto  
  - Hora del viaje
  - Forma de pago
- Un **bucket de GCS** que organiza todo el ciclo de vida del dato:
  - `raw/`, `curated/`, `agg/`, `models/`.
- Una **web app Flask** que:
  - Expone un formulario para capturar los datos del viaje.
  - Consulta el modelo entrenado en GCS.
  - Devuelve una **tarifa estimada** y un pequeño resumen visual.

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
        PySpark ETL en Dataproc (cluster nyc-taxi-etl-pyspark)
                     │
       ┌─────────────┴─────────────┐
       │                           │
  Limpieza + FE               Entrenamiento modelo
                                   │
                                   ▼
                     Modelo GBT guardado en GCS
                                   │
                                   ▼
                      Flask Web App (webapp.py)
                      + model_loader.py
                                   │
                                   ▼
                       UI HTML (templates/index.html)
```

---

## Infraestructura en GCP:

- **Cluster Dataproc:** `nyc-taxi-etl-pyspark` (4 nodos trabajadores, imagen 2.2.70-debian12).
- **Bucket GCS:** `nyc-taxi-etl` con la jerarquía:
    - `raw/nyc_taxi/…`
    - `curated/nyc_taxi/…`
    - `agg/nyc_taxi/…`
    - `models/nyc_taxi_fare_gbt_2015_01/metadata/stages.json`
- **Red privada + Cloud NAT** para que el cluster tenga salida a internet sin IP pública.
- **Acceso desde local** mediante `gcloud` con tunelización de puertos (port-forwarding) para ver la interfaz web.

---

## Estructura del repositorio

```text
nyc-taxi-etl-pyspark/
│
├─ src/
│   ├─ main_etl.py           # Pipeline ETL + entrenamiento del modelo en PySpark
│   ├─ __init__.py
│   └─ api/
│       ├─ webapp.py         # Aplicación Flask
│       ├─ model_loader.py   # Carga del modelo desde GCS y cache en memoria
│       └─ templates/
│           └─ index.html    # Interfaz web (formulario + visualizaciones)
│
├─ .gitignore
└─ README.md                 # Este archivo
```

---

## Tecnologías utilizadas

- **Lenguajes y librerías**

  - Python 3.x

  - PySpark (pyspark.sql, pyspark.ml)

  - Flask

  - Jinja2 (templates HTML)

  - Pandas / NumPy (apoyo ligero)

- **Google Cloud Platform**

  - Dataproc: ejecución distribuida de PySpark.

  - Cloud Storage (GCS): almacenamiento de datos y modelos.

  - VPC + Cloud Router + Cloud NAT: salida a internet sin exponer el cluster.

  - gcloud CLI: tunelización de puertos.

--- 

## Dataset

Se utilizó el dataset público **Yellow Taxi Trip Records** de la NYC TLC.

Para este proyecto se trabajó con un subconjunto representativo:

- Mes: **enero 2015**
- Formato: CSV o Parquet según disponibilidad
- Volumen aproximado: ~12 millones de registros

### Variables más relevantes

- `pickup_datetime`, `dropoff_datetime`
- `pickup_longitude`, `pickup_latitude`
- `dropoff_longitude`, `dropoff_latitude`
- `trip_distance`
- `passenger_count`
- `payment_type`
- `fare_amount` (variable objetivo)
- `tolls_amount`, `extra`, `mta_tax`, etc.

Estas variables se procesaron en el pipeline ETL para generar las características que alimentan el modelo.

--- 

## Pipeline ETL en PySpark (`main_etl.py`)

Resumen de pasos principales:

1. **Configuración del SparkSession** con los conectores necesarios para GCS.

2. **Lectura de datos crudos** desde:  
   `gs://nyc-taxi-etl/raw/nyc_taxi/…`

3. **Limpieza y validación:**
    - Eliminación de filas con coordenadas inválidas.
    - Filtro de tarifas negativas o ridículamente altas.
    - Filtro de distancias nulas o inconsistentes.

4. **Feature Engineering:**
    - Cálculo de la distancia de viaje (aprox. haversine o similar).
    - Extracción de variables de tiempo: hora del día, día de la semana, etc.
    - Codificación de variables categóricas necesarias.

5. **Escritura de la capa curated:**
    - Datos listos para modelado →  
      `gs://nyc-taxi-etl/curated/nyc_taxi/…`

6. **Agregaciones (agg layer):**
    - Métricas por hora, zona, tipo de tarifa, etc. →  
      `gs://nyc-taxi-etl/agg/nyc_taxi/…`

7. **Entrenamiento del modelo:**
    - Uso de **Gradient Boosted Trees Regressor** (`GBTRegressor`).
    - Pipeline de ML con VectorAssembler + normalización.
    - Evaluación con RMSE y MAE (impresos en logs).

8. **Persistencia del modelo:**
    - El mejor modelo y su pipeline se guardan en:  
      `gs://nyc-taxi-etl/models/nyc_taxi_fare_gbt_2015_01/metadata/stages.json`
    - Subcarpeta `stages/` con los artefactos de Spark ML.


---

## Carga del modelo y API web

### `model_loader.py`

Archivo auxiliar pensado para centralizar la lógica de carga de modelos Spark ML desde GCS.  
En esta versión del proyecto, la carga del modelo se realiza directamente en `webapp.py`, pero `model_loader.py` queda preparado para futuras refactorizaciones.


### `webapp.py`

El modelo se carga directamente desde GCS:

```python
from pyspark.ml import PipelineModel

MODEL_PATH = "gs://nyc-taxi-etl/models/nyc_taxi_fare_gbt_2015_01"
model = PipelineModel.load(MODEL_PATH)

```

- Web app con **Flask**.
- Rutas principales:
    - **GET /**  
      Renderiza `templates/index.html` con el formulario para capturar datos del viaje.
    - **POST /predict_web**  
      Toma los campos del formulario, construye un DataFrame Spark, llama al modelo y devuelve:
        - Tarifa estimada.          
- Está pensada para ejecutarse sobre el cluster / VM y ser consumida desde el navegador mediante un túnel con `gcloud`.

--- 

## Interfaz web (`templates/index.html`)

La interfaz gráfica fue diseñada para proporcionar una experiencia sencilla y moderna inspirada en la estética de los **yellow cabs** de NYC.

Incluye:

### Formulario de entrada
El usuario ingresa:

- Distancia del viaje (millas)
- Cantidad de pasajeros
- Hora del día (0–23)
- Tipo de pago (Credit Card, Cash, etc.)

El formulario envía una petición POST a `/predict_web`.

### Visualización de resultados

Una vez procesada la predicción, la UI presenta:

#### 1. Tarifa estimada
Se muestra en un estilo grande y destacado (`$XX.XX`).

#### 2. Resumen del viaje
Pequeñas *chips* visuales muestran:

- Distancia ingresada  
- Número de pasajeros  
- Hora del día  
- Duración estimada (calculada en la app)

#### 3. Velocímetro (Gauge) con Chart.js
Visualización semicircular que representa:

- El valor estimado (arco amarillo)
- El resto del rango (arco gris)

Implementado con **Chart.js (doughnut chart)** configurado como gauge.

#### 4. Mapa ilustrativo (Leaflet)
Un mapa interactivo centrado en Manhattan utilizando:

```js
L.map('map').setView([40.75, -73.98], 11);
```
Esto no representa el trayecto real, sino una referencia de zona geográfica típica de operación de taxis amarillos.

---  

## Infraestructura en Google Cloud

### 1. Bucket de GCS

Bucket: `nyc-taxi-etl` (región `us-central1`)

```text
nyc-taxi-etl/
├─ raw/
│  └─ nyc_taxi/…              # Datos crudos
├─ curated/
│  └─ nyc_taxi/…              # Datos limpios
├─ agg/
│  └─ nyc_taxi/…              # Tablas agregadas para analytics
└─ models/
   └─ nyc_taxi_fare_gbt_2015_01/
      ├─ metadata/stages.json # Configuración del modelo
      └─ stages/              # Artefactos de Spark ML
```

### 2. Cluster Dataproc

 - Nombre: nyc-taxi-etl-pyspark

 - Región: us-central1

 - Imagen base: 2.2.70-debian12

 - Topología: 1 nodo master + 4 workers

### 3. Red y NAT

  - VPC privada sin IP pública directa en los nodos.

  - Cloud NAT configurado para permitir:

      - Descarga de paquetes.

      - Acceso a repositorios.

  - El acceso desde la máquina local se hizo con:

      - gcloud dataproc jobs submit pyspark para correr main_etl.py.

      - Tunelización de puertos para exponer:

          - Spark Web UI (puerto 4040/8088, etc.).

          - Flask web app (webapp.py) mapeada a un puerto local.

---

## Roadmap / Trabajo futuro

- Implementar endpoint JSON (`/predict`) para consumo programático.
- Guardar métricas de entrenamiento en un sistema de tracking (MLflow, BigQuery, etc.).
- Añadir validaciones más robustas en la web app.
- Desplegar la API en Cloud Run o GKE.
- Añadir tests unitarios para las funciones clave del ETL.
