# main.py
import streamlit as st
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS


st.set_page_config(page_title="Weather & Power – IND320", layout="wide")

# sensible defaults in session
if "price_area" not in st.session_state:
    st.session_state.price_area = "NO1"
if "month" not in st.session_state:
    st.session_state.month = 1
if "groups" not in st.session_state:
    st.session_state.groups = ["hydro","wind","thermal","solar","other"]

st.title("IND320 – Weather & Production App")

st.markdown("""
Choose area and moth. These will be set as default selections on the next pages.
""")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.session_state.price_area = st.selectbox("Price area",
        ["NO1","NO2","NO3","NO4","NO5"], index=0)
with col2:
    st.session_state.month = st.select_slider("Month", options=list(range(1,13)), value=1)
with col3:
    all_groups = ["hydro","wind","thermal","solar","other"]
    st.session_state.groups = st.pills("Production groups",
        options=all_groups, default=all_groups, selection_mode="multi")

st.success("Default selections saved i session. Go to ´Production (Elhub)´to view the graphs")
