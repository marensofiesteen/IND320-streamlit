import streamlit as st
import pandas as pd
from pymongo import MongoClient

st.set_page_config(page_title="IND320 – Elhub", layout="wide")

# --- Config
cfg = st.secrets["mongo"]
client = MongoClient(cfg["uri"])
col = client[cfg["db"]][cfg["collection"]]

# --- Fetch data (project only the fields we need)
@st.cache_data(show_spinner=False, ttl=600)
def load_data(limit=None):
    cursor = col.find(
        {},
        {"_id": 0, "priceArea": 1, "productionGroup": 1, "quantityKwh": 1, "startTime": 1},
        sort=[("startTime", 1)]
    )
    if limit:
        cursor = cursor.limit(limit)
    df = pd.DataFrame(list(cursor))

    # Typing
    if not df.empty:
        df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce", utc=True).dt.tz_convert("Europe/Oslo")
        df["quantityKwh"] = pd.to_numeric(df["quantityKwh"], errors="coerce")
        df["priceArea"] = df["priceArea"].astype("string")
        df["productionGroup"] = df["productionGroup"].astype("string")
        df = df.dropna(subset=["startTime"])  # ensure the time axis
    return df

with st.spinner("Laster data fra Atlas…"):
    df = load_data()  # evt. load_data(limit=200000) first time

st.success(f"Data OK: {len(df):,} rader • Kolonner: {list(df.columns)}")

# --- Simple filters in UI
left, right = st.columns(2)
with left:
    areas = sorted(df["priceArea"].dropna().unique().tolist())
    sel_area = st.multiselect("Price area", areas, default=areas[:1] if areas else [])
with right:
    groups = sorted(df["productionGroup"].dropna().unique().tolist())
    sel_group = st.multiselect("Production group", groups, default=groups[:3] if groups else [])

mask = pd.Series(True, index=df.index)
if sel_area:
    mask &= df["priceArea"].isin(sel_area)
if sel_group:
    mask &= df["productionGroup"].isin(sel_group)
view = df.loc[mask].copy()

# Example: aggregate per day
daily = (
    view.assign(date=view["startTime"].dt.date)
        .groupby(["date", "priceArea", "productionGroup"], as_index=False)["quantityKwh"]
        .sum()
)
st.write("Daily aggregate (first 10):")
st.dataframe(daily.head(10))
