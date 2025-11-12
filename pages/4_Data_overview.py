import streamlit as st
import pandas as pd
from db_mongo import fetch_month
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS

st.set_page_config(page_title="Data overview", layout="wide")
st.title("Data overview")

# -----------------------------
# Elhub Production (MongoDB)
# -----------------------------
st.header("Elhub Production (from MongoDB)")

pa = st.selectbox("Price area (Elhub)", ["NO1","NO2","NO3","NO4","NO5"],
                  index=["NO1","NO2","NO3","NO4","NO5"].index(st.session_state.get("price_area","NO1")))
mo = st.select_slider("Month", options=list(range(1,13)),
                      value=int(st.session_state.get("month",1)))

df_elhub = fetch_month(pa, ["hydro","wind","thermal","solar","other"], int(mo))
st.dataframe(df_elhub.head(200), use_container_width=True)
st.caption(f"Rows (selected month): {len(df_elhub):,}")

# -----------------------------
# Open-Meteo Hourly (API, 2021)
# -----------------------------
st.header("Open-Meteo Hourly (API, 2021)")

pa2 = st.selectbox("Price area (weather)", list(PRICE_AREAS.keys()),
                   index=list(PRICE_AREAS.keys()).index(st.session_state.get("price_area","NO1")))
df_om = fetch_openmeteo_hourly(pa2, 2021)

if df_om.empty:
    st.warning("No data from Open-Meteo (check API).")
else:
    st.dataframe(df_om.head(200), use_container_width=True)
    st.caption(f"Rows (full year): {len(df_om):,}")
