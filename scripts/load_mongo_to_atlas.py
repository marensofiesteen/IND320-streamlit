import pandas as pd
from pymongo import MongoClient
import time
import tomllib, pathlib

# --- Read in secrets ---
cfg = tomllib.loads(pathlib.Path(".streamlit/secrets.toml").read_text())
uri = cfg["mongo"]["uri"]
db_name = cfg["mongo"]["db"]
collection_name = cfg["mongo"]["collection"]

# --- Read CSV ---
df = pd.read_csv("data/Elhub_production_2021_all_areas.csv")
print("✅ CSV loaded with columns:", list(df.columns))

# --- Connect to MongoDB Atlas ---
client = MongoClient(uri)
db = client[db_name]
collection = db[collection_name]

# --- Delete existing data (optional) ---
collection.drop()
print(f"🗑️ Dropped existing collection '{collection_name}' in the database '{db_name}'")

# --- Convert to dictionary and load up in batcher ---
records = df.to_dict(orient="records")
batch_size = 10000
total = len(records)

for i in range(0, total, batch_size):
    collection.insert_many(records[i:i + batch_size])
    print(f"✅ Inserted {min(i+batch_size, total)}/{total}")
    time.sleep(0.5)

print("🎯 Finished! Number of documents in MongoDB Atlas:", collection.count_documents({}))
