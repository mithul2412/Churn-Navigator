import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col

# Check if JAVA_HOME is set
if 'JAVA_HOME' not in os.environ:
    print("JAVA_HOME is not set. Setting it now...")
    # Try to find Java and set it
    try:
        import subprocess
        java_path = subprocess.check_output(['readlink', '-f', '/usr/bin/java']).decode('utf-8').strip()
        java_home = java_path.replace('/bin/java', '')
        os.environ['JAVA_HOME'] = java_home
        print(f"Set JAVA_HOME to {java_home}")
    except Exception as e:
        print(f"Failed to automatically set JAVA_HOME: {e}")
        print("Please set JAVA_HOME manually and try again.")
        sys.exit(1)

# Create Spark session with MongoDB connector
spark = SparkSession.builder \
    .appName("ChurnFeatureTransformation") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()

# MongoDB connection URI
mongodb_uri = "mongodb+srv://churn:churn@telco.l9welj0.mongodb.net/churn"
collection_name = "churn"

# Read data from MongoDB
df = spark.read.format("mongo") \
    .option("uri", mongodb_uri) \
    .option("collection", collection_name) \
    .load()

# Display schema and sample data
print("Schema:")
df.printSchema()
print("\nSample Data:")
df.show(5)

# Feature preprocessing
# 1. Convert categorical features to numeric using StringIndexer
categorical_cols = ["gender", "Partner", "Dependents", "PhoneService", 
                   "MultipleLines", "InternetService", "OnlineSecurity", 
                   "OnlineBackup", "DeviceProtection", "TechSupport", 
                   "StreamingTV", "StreamingMovies", "Contract", 
                   "PaperlessBilling", "PaymentMethod"]

# Create indexers for each categorical column
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep") 
            for col in categorical_cols]

# 2. Apply one-hot encoding to indexed categories
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded") 
            for col in categorical_cols]

# 3. Prepare numerical features
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# Create feature vector by combining all features
encoded_cols = [col+"_encoded" for col in categorical_cols]
assembler_inputs = encoded_cols + numeric_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

# 4. Target variable transformation (Churn → 0/1)
label_indexer = StringIndexer(inputCol="Churn", outputCol="label", handleInvalid="keep")

# Build the pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, label_indexer])

# Apply transformations
transformed_data = pipeline.fit(df).transform(df)

# Keep only necessary columns for ML
final_data = transformed_data.select("customerID", "features", "label")

# Save the transformed data
# output_path = "data/transformed_churn_data"
output_path = "/home/stlp/spark_output/transformed_churn_data"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_data.write.parquet(output_path, mode="overwrite")

print(f"Transformed data saved to {output_path}")
spark.stop()