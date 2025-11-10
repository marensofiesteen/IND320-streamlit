import streamlit as st

st.set_page_config(page_title="Area & Year", layout="wide")
st.title("Area & Year Selector")

areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
sel_area = st.selectbox("Select price area", areas, index=0)
sel_year = st.selectbox("Select year", list(range(2018, 2025)), index=areas.index("NO1"))

st.session_state["selected_area"] = sel_area
st.session_state["selected_year"] = sel_year

st.success(f"Stored: area = **{sel_area}**, year = **{sel_year}** in session_state.")
