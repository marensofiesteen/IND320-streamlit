import streamlit as st
import pandas as pd

from db_mongo import fetch_month
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS

st.set_page_config(page_title="Data overview", layout="wide")
st.title("Data overview")

# -----------------------------
# Elhub Production (MongoDB)
# -----------------------------
st.header("Elhub production (from MongoDB)")

elhub_areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]

default_area = st.session_state.get("price_area", "NO1")
if default_area not in elhub_areas:
    default_area = "NO1"

default_month = int(st.session_state.get("month", 1))
if default_month < 1 or default_month > 12:
    default_month = 1

pa = st.selectbox(
    "Price area (Elhub)",
    elhub_areas,
    index=elhub_areas.index(default_area),
)
st.session_state["price_area"] = pa

mo = st.select_slider(
    "Month",
    options=list(range(1, 13)),
    value=default_month,
)
st.session_state["month"] = int(mo)

df_elhub = fetch_month(pa, ["hydro", "wind", "thermal", "solar", "other"], int(mo))
st.dataframe(df_elhub.head(200), use_container_width=True)
st.caption(f"Rows (selected month): {len(df_elhub):,}")

# -----------------------------
# Open-Meteo Hourly (API, 2021)
# -----------------------------
st.header("Open-Meteo hourly (API, 2021)")

weather_areas = list(PRICE_AREAS.keys())

default_weather_area = st.session_state.get("price_area", "NO1")
if default_weather_area not in weather_areas:
    default_weather_area = "NO1"

pa2 = st.selectbox(
    "Price area (weather)",
    weather_areas,
    index=weather_areas.index(default_weather_area),
)
# Keep global price area in sync (using the latest choice)
st.session_state["price_area"] = pa2

df_om = fetch_openmeteo_hourly(pa2, 2021)

if df_om.empty:
    st.warning("No data from Open-Meteo (check API).")
else:
    st.dataframe(df_om.head(200), use_container_width=True)
    st.caption(f"Rows (full year): {len(df_om):,}")
