import argparse
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

try:
    from config import (
        MONGO_COLLECTION,
        MONGO_DB,
        MONGO_URI,
        SPARK_JARS_PACKAGES,
        SPARK_OUTPUT_PATH,
    )
except ImportError:
    from scripts.config import (
        MONGO_COLLECTION,
        MONGO_DB,
        MONGO_URI,
        SPARK_JARS_PACKAGES,
        SPARK_OUTPUT_PATH,
    )

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_spark_session(mongo_uri: str) -> SparkSession:
    builder = SparkSession.builder.appName("ChurnFeatureTransformation")
    if SPARK_JARS_PACKAGES:
        builder = builder.config("spark.jars.packages", SPARK_JARS_PACKAGES)

    return (
        builder.config("spark.mongodb.read.connection.uri", mongo_uri)
        .config("spark.mongodb.write.connection.uri", mongo_uri)
        .getOrCreate()
    )


def transform_data(spark: SparkSession, mongo_db: str, mongo_collection: str):
    df = (
        spark.read.format("mongodb")
        .option("spark.mongodb.read.database", mongo_db)
        .option("spark.mongodb.read.collection", mongo_collection)
        .load()
    )

    for numeric_col in NUMERIC_COLS:
        df = df.withColumn(numeric_col, col(numeric_col).cast("double"))

    df = df.fillna({"TotalCharges": 0.0, "MonthlyCharges": 0.0, "tenure": 0.0})

    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_index", handleInvalid="keep")
        for column in CATEGORICAL_COLS
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{column}_index", outputCol=f"{column}_encoded")
        for column in CATEGORICAL_COLS
    ]

    assembler_inputs = [f"{column}_encoded" for column in CATEGORICAL_COLS] + NUMERIC_COLS
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

    label_indexer = StringIndexer(inputCol="Churn", outputCol="label", handleInvalid="keep")
    pipeline = Pipeline(stages=indexers + encoders + [assembler, label_indexer])

    transformed = pipeline.fit(df).transform(df)
    return transformed.select(
        "customerID",
        vector_to_array(col("features")).alias("features"),
        col("label").cast("int").alias("label"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform churn data from MongoDB into ML-ready features")
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    parser.add_argument("--mongo-db", default=MONGO_DB)
    parser.add_argument("--mongo-collection", default=MONGO_COLLECTION)
    parser.add_argument("--output-path", default=SPARK_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spark = build_spark_session(args.mongo_uri)
    try:
        features_df = transform_data(spark, args.mongo_db, args.mongo_collection)
        features_df.write.mode("overwrite").parquet(str(output_path))
        print(f"Transformed feature data saved to {output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
