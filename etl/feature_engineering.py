import argparse

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark ETL for churn feature engineering")
    parser.add_argument("--mongo-uri", required=True)
    parser.add_argument("--mongo-db", default="churn")
    parser.add_argument("--mongo-collection", default="customer_churn")
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--test-out", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--spark-packages",
        default="org.mongodb.spark:mongo-spark-connector_2.12:10.3.0",
    )
    return parser.parse_args()


def build_spark(args: argparse.Namespace) -> SparkSession:
    builder = SparkSession.builder.appName("ChurnFeatureEngineering")
    if args.spark_packages:
        builder = builder.config("spark.jars.packages", args.spark_packages)

    return (
        builder.config("spark.mongodb.read.connection.uri", args.mongo_uri)
        .config("spark.mongodb.write.connection.uri", args.mongo_uri)
        .getOrCreate()
    )


def run_feature_engineering(spark: SparkSession, args: argparse.Namespace):
    raw_df = (
        spark.read.format("mongodb")
        .option("spark.mongodb.read.database", args.mongo_db)
        .option("spark.mongodb.read.collection", args.mongo_collection)
        .load()
    )

    for numeric_col in NUMERIC_COLS:
        raw_df = raw_df.withColumn(numeric_col, col(numeric_col).cast("double"))

    raw_df = raw_df.fillna({"TotalCharges": 0.0, "MonthlyCharges": 0.0, "tenure": 0.0})

    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_index", handleInvalid="keep")
        for column in CATEGORICAL_COLS
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{column}_index", outputCol=f"{column}_encoded")
        for column in CATEGORICAL_COLS
    ]
    assembler_inputs = [f"{column}_encoded" for column in CATEGORICAL_COLS] + NUMERIC_COLS

    pipeline = Pipeline(
        stages=indexers
        + encoders
        + [
            VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep"),
            StringIndexer(inputCol="Churn", outputCol="label", handleInvalid="keep"),
        ]
    )

    transformed = pipeline.fit(raw_df).transform(raw_df).select(
        "customerID",
        vector_to_array(col("features")).alias("features"),
        col("label").cast("int").alias("label"),
    )

    train_df, test_df = transformed.randomSplit([1.0 - args.test_fraction, args.test_fraction], seed=42)
    train_df.write.mode("overwrite").parquet(args.train_out)
    test_df.write.mode("overwrite").parquet(args.test_out)

    print(f"Train rows: {train_df.count()} -> {args.train_out}")
    print(f"Test rows: {test_df.count()} -> {args.test_out}")


def main() -> None:
    args = parse_args()
    spark = build_spark(args)
    try:
        run_feature_engineering(spark, args)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
