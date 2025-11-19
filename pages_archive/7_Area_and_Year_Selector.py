import streamlit as st
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS

st.set_page_config(page_title="Area & Year", layout="wide")
st.title("Area & Year Selector")

areas = ["NO1","NO2","NO3","NO4","NO5"]
years = [2021]  # del 3 krever 2021

sel_area = st.selectbox("Select price area", areas,
                        index=areas.index(st.session_state.get("price_area","NO1")))
sel_year = st.selectbox("Select year", years, index=0)

st.session_state["price_area"] = sel_area
st.session_state["selected_year"] = sel_year
st.success(f"Stored: area = **{sel_area}**, year = **{sel_year}**.")
