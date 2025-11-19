from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql import functions as F

"""
ETL script: loads Elhub CSV into Cassandra (keyspace 'elnub', table 'production_data').

This script is used during data preparation and is not called by the Streamlit app.
"""

# --- Start Spark with Cassandra-connector ---
spark = (
    SparkSession.builder
    .appName("csv-to-cassandra")
    .master("local[*]")
    # Viktig: connector til Cassandra
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1")
    .config("spark.sql.extensions", "com.datastax.spark.connector.CassandraSparkExtensions")
    .config("spark.sql.catalog.mycatalog", "com.datastax.spark.connector.datasource.CassandraCatalog")
    # Nettverksoppsett lokalt
    .config("spark.driver.bindAddress","127.0.0.1")
    .config("spark.driver.host","127.0.0.1")
    # Hvor Cassandra kjører
    .config("spark.cassandra.connection.host","127.0.0.1")
    .config("spark.cassandra.connection.port","9042")
    .getOrCreate()
)

# --- Read CSV robust (auto-detekt ; vs ,) ---
path = "data/Elhub_production_2021_all_areas.csv"

# Find delimiter from header
with open(path, "r", encoding="utf-8") as f:
    header_line = f.readline().strip()
sep = ";" if header_line.count(";") > header_line.count(",") else ","
print("Using separator:", sep)

sdf = (
    spark.read
    .option("header", True)
    .option("sep", sep)           # <- must be string
    .option("encoding", "utf-8")
    .csv(path)
)

# If we still got a single column, try the opposite separator
if len(sdf.columns) == 1 and any(x in sdf.columns[0] for x in [",",";"]):
    alt_sep = "," if sep == ";" else ";"
    print("Retry with separator:", alt_sep)
    sdf = (
        spark.read
        .option("header", True)
        .option("sep", alt_sep)
        .option("encoding", "utf-8")
        .csv(path)
    )


# --- Type casting + cleanup ---
# Select and assign correct names (lowercase) + data types
sdf_norm = (sdf
    .select(
        F.col("priceArea").alias("pricearea"),
        F.col("productionGroup").alias("productiongroup"),
        F.to_timestamp("startTime").alias("starttime"),
        F.col("quantityKwh").cast("double").alias("quantitykwh"),
        F.to_timestamp("endTime").alias("endtime"),
        F.to_timestamp("lastUpdatedTime").alias("lastupdatedtime"),
    )
)

sdf_norm.printSchema()
sdf_norm.show(5, truncate=False)

# Write to Cassandra (must matche keyspace/tabell over)
(sdf_norm.write
    .format("org.apache.spark.sql.cassandra")
    .options(table="production_data", keyspace="elnub")
    .mode("append")
    .save())

# --- Diagnosis: chech nulls in the primary key---
from pyspark.sql import functions as F

pk_nulls = (sdf_norm
    .select(
        F.sum(F.col("pricearea").isNull().cast("int")).alias("null_pricearea"),
        F.sum(F.col("productiongroup").isNull().cast("int")).alias("null_productiongroup"),
        F.sum(F.col("starttime").isNull().cast("int")).alias("null_starttime"),
    )
    .collect()[0]
)
print("[PK NULLS]", pk_nulls)

# Drop rows missing primary key columns
sdf_clean = sdf_norm.dropna(subset=["pricearea", "productiongroup", "starttime"])

# (optional) count how many are dropped
before = sdf_norm.count()
after  = sdf_clean.count()
print(f"[CLEAN] kept {after:,} / {before:,} rows (dropped {before-after:,})")

# --- Write to Cassandra ---
(sdf_clean.write
 .format("org.apache.spark.sql.cassandra")
 .options(table="production_data", keyspace="elnub")
 .mode("append")
 .save())

print("✅ Wrote to Cassandra")
