import pandas as pd
from pymongo import MongoClient
import streamlit as st
from datetime import datetime, timezone

@st.cache_resource
def get_mongo_client():
    return MongoClient(st.secrets["mongo"]["uri"])

def _col():
    cli = get_mongo_client()
    return cli[st.secrets["mongo"]["db"]][st.secrets["mongo"]["collection"]]

def _month_bounds(year: int, month: int):
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return start, end

@st.cache_data(show_spinner=False, ttl=600)
def fetch_month(price_area: str, groups: list[str], month: int) -> pd.DataFrame:
    col = _col()
    start, end = _month_bounds(2021, int(month))

    q = {
        "priceArea": price_area,
        "productionGroup": {"$in": groups},
        "startTime": {"$gte": start, "$lt": end},   # bruker indeks vennlig datointervall
    }

    proj = {"_id": 0, "priceArea": 1, "productionGroup": 1, "startTime": 1, "quantityKwh": 1}
    docs = list(col.find(q, proj))  # ingen server-side sort -> sorter i pandas

    if not docs:
        return pd.DataFrame(columns=["pricearea","productiongroup","datetime","quantitykwh"])

    df = pd.DataFrame(docs).rename(columns={
        "priceArea": "pricearea",
        "productionGroup": "productiongroup",
        "startTime": "datetime",
        "quantityKwh": "quantitykwh",
    })

    # Datatime objects are already datetime instances; ensure UTC and optionally display local time in the app.
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    # If you want to show local Norwegian time in the chart:
    # df["datetime"] = df["datetime"].dt.tz_convert("Europe/Oslo")

    df["quantitykwh"] = pd.to_numeric(df["quantitykwh"], errors="coerce")
    df = df.dropna(subset=["datetime", "quantitykwh"]).sort_values("datetime").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False, ttl=600)
def fetch_year(price_area: str, production_group: str) -> pd.DataFrame:
    parts = [fetch_month(price_area, [production_group], m) for m in range(1, 13)]
    if not any(len(p) for p in parts):
        return pd.DataFrame(columns=["pricearea","productiongroup","datetime","quantitykwh"])
    return pd.concat(parts, ignore_index=True)
