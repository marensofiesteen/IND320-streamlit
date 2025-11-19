import streamlit as st
import pandas as pd
import plotly.express as px

from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS

st.set_page_config(page_title="Weather plots", layout="wide")
st.header("Time series exploration of weather data (Open-Meteo API, 2021)")

# --- Global price area selection ---
weather_areas = list(PRICE_AREAS.keys())

default_area = st.session_state.get("price_area", "NO1")
if default_area not in weather_areas:
    default_area = "NO1"

pa = st.selectbox(
    "Price area",
    weather_areas,
    index=weather_areas.index(default_area),
)

# Keep global selection in sync so other pages see the same area
st.session_state["price_area"] = pa

df = fetch_openmeteo_hourly(pa, 2021)
if df.empty:
    st.error("No data from Open-Meteo.")
    st.stop()

df["year_month"] = df["date"].dt.strftime("%Y-%m")

exclude = {"date", "year_month", "priceArea", "city"}
meteo_options = [c for c in df.columns if c not in exclude]

option_meteo = st.selectbox(
    "Step 1: Select one variable",
    options=meteo_options + ["Show all"],
    index=0,
)

month_list = sorted(df["year_month"].dropna().unique().tolist())
option_month = st.select_slider(
    "Step 2: Select a month",
    options=month_list,
    value=month_list[0],
)

df_filtered = df[df["year_month"] <= option_month].copy()
start_month = df_filtered["year_month"].min()

if option_meteo != "Show all":
    fig = px.line(
        df_filtered,
        x="date",
        y=option_meteo,
        title=f"{option_meteo} from {start_month} to {option_month}",
    )
else:
    df_melt = df_filtered.melt(
        id_vars=["date"],
        value_vars=meteo_options,
        var_name="variable",
        value_name="value",
    )
    fig = px.line(
        df_melt,
        x="date",
        y="value",
        color="variable",
        title=f"All variables from {start_month} to {option_month}",
    )

st.plotly_chart(fig, use_container_width=True)

