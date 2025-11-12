# load_mongo_from_csv.py
import pandas as pd
from pymongo import MongoClient, ASCENDING
import tomllib, pathlib

# --- secrets ---
cfg = tomllib.loads(pathlib.Path(".streamlit/secrets.toml").read_text())
URI = cfg["mongo"]["uri"]
DB = cfg["mongo"]["db"]
COL = cfg["mongo"]["collection"]

# --- read CSV robust: auto-sep + BOM ---
path = "data/Elhub_production_2021_all_areas.csv"
df = pd.read_csv(
    path,
    sep=None,              # auto-detect delimiter (comma, semicolon, tab, etc.)
    engine="python",       # required for sep=None
    encoding="utf-8-sig",  # handles BOM if present
)
print("DEBUG columns read:", list(df.columns))


# --- clean columnnames (strip + lower) and map to standard ---
orig_cols = {c.lower().strip(): c for c in df.columns}
need_map = {
    "pricearea": "priceArea",
    "productiongroup": "productionGroup",
    "quantitykwh": "quantityKwh",
    "starttime": "startTime",
}

# check that all are present, otherwise raise a clear error message.
missing = [k for k in need_map if k not in orig_cols]
if missing:
    raise KeyError(f"Mangler forventede kolonner i CSV (case-insensitivt): {missing}\nFikk: {list(df.columns)}")

# build a “safe” view with the correct original names
df_safe = df[[orig_cols["pricearea"], orig_cols["productiongroup"], orig_cols["quantitykwh"], orig_cols["starttime"]]].copy()
df_safe.columns = ["priceArea", "productionGroup", "quantityKwh", "startTime"]

# types
df_safe["quantityKwh"] = pd.to_numeric(df_safe["quantityKwh"], errors="coerce")
dt = pd.to_datetime(df_safe["startTime"], errors="coerce", utc=True)
# to naive UTC (BSON Date)
df_safe["startTime"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)

# skip empty
df_safe = df_safe.dropna(subset=["priceArea", "productionGroup", "quantityKwh", "startTime"])

print("DEBUG preview:\n", df_safe.head())

# --- inn i Mongo ---
cli = MongoClient(URI)
col = cli[DB][COL]

# col.delete_many({})  # uncomment if you want to start over

docs = df_safe.to_dict(orient="records")
BATCH = 50_000
for i in range(0, len(docs), BATCH):
    col.insert_many(docs[i:i+BATCH])
    print(f"Inserted {min(i+BATCH, len(docs))}/{len(docs)}")

col.create_index(
    [("priceArea", ASCENDING), ("productionGroup", ASCENDING), ("startTime", ASCENDING)],
    background=True
)

print("✅ Finished! Number of documents in Mongo:", col.estimated_document_count())
