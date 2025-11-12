import argparse
from typing import Iterable

import pandas as pd
from pymongo import MongoClient

try:
    from config import DATA_PATH, MONGO_COLLECTION, MONGO_DB, MONGO_URI
except ImportError:
    from scripts.config import DATA_PATH, MONGO_COLLECTION, MONGO_DB, MONGO_URI


def load_churn_dataframe(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def upload_to_mongodb(data_path: str, drop_existing: bool = False) -> int:
    df = load_churn_dataframe(data_path)

    client = MongoClient(MONGO_URI)
    collection = client[MONGO_DB][MONGO_COLLECTION]

    if drop_existing:
        collection.delete_many({})

    records: Iterable[dict] = df.to_dict("records")
    result = collection.insert_many(list(records))

    collection.create_index("customerID")
    collection.create_index("Churn")

    return len(result.inserted_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload churn CSV data to MongoDB")
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to churn dataset CSV")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Clear existing records in the target collection before upload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inserted = upload_to_mongodb(args.data_path, drop_existing=args.drop_existing)
    print(f"Uploaded {inserted} rows to {MONGO_DB}.{MONGO_COLLECTION}")


if __name__ == "__main__":
    main()
