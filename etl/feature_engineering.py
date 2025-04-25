# etl/feature_engineering.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, datediff, current_date
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

if __name__ == "__main__":
    # ────────────────────────────────────────────────────────────────
    # 1) Initialize SparkSession with MongoDB connector
    #
    # We point Spark at your raw and processed collections in Mongo.
    spark = (
        SparkSession.builder
          .appName("ChurnFeatureEngineering")
          # URI for reading raw data
          .config("spark.mongodb.read.connection.uri",
                  "mongodb://admin:changeme@mongo:27017/churndb.raw_churn_data?authSource=admin")
          # URI for writing processed features
          .config("spark.mongodb.write.connection.uri",
                  "mongodb://admin:changeme@mongo:27017/churndb.processed_churn_features?authSource=admin")
          .getOrCreate()
    )

    # ────────────────────────────────────────────────────────────────
    # 2) Load your raw churn DataFrame from MongoDB
    raw_df = spark.read.format("mongo").load()

    # ────────────────────────────────────────────────────────────────
    # 3) Clean + type‐cast TotalCharges
    #    - Empty string → NULL → cast to double → fill NULL with 0.0
    clean_df = (
      raw_df
      .withColumn("TotalCharges",
          when(col("TotalCharges") == "", None).otherwise(col("TotalCharges"))
      )
      .withColumn("TotalCharges", col("TotalCharges").cast("double"))
      .na.fill({"TotalCharges": 0.0})
    )

    # ────────────────────────────────────────────────────────────────
    # 4) Engineer a numeric feature: avgMonthlyCharge
    #    - guard against division by zero when tenure=0
    feature_df = clean_df.withColumn(
      "avgMonthlyCharge",
      when(col("tenure") == 0, 0.0).otherwise(col("TotalCharges") / col("tenure"))
    )

    # ────────────────────────────────────────────────────────────────
    # 5) String‐index your categorical columns
    cats = [
      "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
      "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
      "TechSupport", "StreamingTV", "StreamingMovies",
      "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    indexers = [
      StringIndexer(inputCol=c, outputCol=f"{c}_Idx", handleInvalid="keep")
      for c in cats
    ]

    # ────────────────────────────────────────────────────────────────
    # 6) Assemble all features into a single vector called "features"
    numeric_feats = ["tenure", "MonthlyCharges", "avgMonthlyCharge"]
    assembler = VectorAssembler(
      inputCols=numeric_feats + [f"{c}_Idx" for c in cats],
      outputCol="features"
    )

    # ────────────────────────────────────────────────────────────────
    # 7) Build + run the Pipeline (all indexers → assembler)
    pipeline = Pipeline(stages=indexers + [assembler])
    model    = pipeline.fit(feature_df)
    output   = model.transform(feature_df)

    # ────────────────────────────────────────────────────────────────
    # 8) Add your label column: 1.0 if Churn == "Yes", else 0.0
    from pyspark.sql.functions import lit
    labeled = output.withColumn(
      "label",
      when(col("Churn") == "Yes", lit(1.0)).otherwise(lit(0.0))
    )

    # ────────────────────────────────────────────────────────────────
    # 9) Write your processed feature table back to MongoDB
    (
      labeled
      .select("customerID", "features", "label")   # pick the columns you need
      .write
      .format("mongo")
      .mode("overwrite")                           # replaces prior run
      .save()
    )

    spark.stop()
