from pyspark.sql import functions as F, types as T
from pyspark.sql.functions import broadcast

def clean_and_transform(df_raw):
    df = (
        df_raw
        .withColumn("tpep_pickup_datetime", F.col("tpep_pickup_datetime").cast("timestamp"))
        .withColumn("tpep_dropoff_datetime", F.col("tpep_dropoff_datetime").cast("timestamp"))
    )

    df = df.withColumn(
        "trip_duration_min",
        (F.col("tpep_dropoff_datetime").cast("long") - 
         F.col("tpep_pickup_datetime").cast("long")) / 60.0
    )

    df = df.filter("trip_distance > 0 AND fare_amount > 0 AND total_amount > 0 AND passenger_count > 0")
    df = df.filter("trip_duration_min BETWEEN 1 AND 180")

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

    df = df.withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
    df = df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    df = df.withColumn("pickup_dow", F.date_format("tpep_pickup_datetime", "E"))

    df = df.withColumn(
        "avg_speed_kmh",
        F.col("trip_distance") / (F.col("trip_duration_min") / 60.0)
    ).filter("avg_speed_kmh BETWEEN 0 AND 120")

    payment_lookup = df.sparkSession.createDataFrame(
        [(1, "Credit card"), (2, "Cash"), (3, "No charge"),
         (4, "Dispute"), (5, "Unknown"), (6, "Voided trip")],
        ["payment_type", "payment_desc"]
    )

    df = df.join(broadcast(payment_lookup), "payment_type", "left")

    return df
