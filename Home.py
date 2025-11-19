import streamlit as st

st.set_page_config(page_title="IND320 – Weather & Production", layout="wide")


# --- Global defaults stored in session_state ---
if "price_area" not in st.session_state:
    st.session_state["price_area"] = "NO1"

if "selected_year" not in st.session_state:
    st.session_state["selected_year"] = 2021  # project uses 2021

if "month" not in st.session_state:
    st.session_state["month"] = 1

if "groups" not in st.session_state:
    st.session_state["groups"] = ["hydro", "wind", "thermal", "solar", "other"]

st.title("IND320 – Weather & Production Dashboard")

st.markdown(
    """
This page defines the **global selection** for the app:

- Price area  
- Year (Elhub production)  
- Month (for weather views, if applicable)  
- Production groups  

These choices are stored in `st.session_state` and used as default values
on the other pages (Elhub production, weather, STL & spectrogram, anomalies, etc.).
"""
)

areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
years = [2021]  # project specification: 2021

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    # Price area
    current_area = st.session_state.get("price_area", "NO1")
    if current_area not in areas:
        current_area = "NO1"

    area = st.selectbox(
        "Price area",
        areas,
        index=areas.index(current_area),
    )
    st.session_state["price_area"] = area

with col2:
    # Year and month
    current_year = st.session_state.get("selected_year", 2021)
    year = st.selectbox("Year", years, index=years.index(current_year))
    st.session_state["selected_year"] = year

    current_month = st.session_state.get("month", 1)
    month = st.select_slider(
        "Month",
        options=list(range(1, 13)),
        value=current_month,
    )
    st.session_state["month"] = month

with col3:
    # Production groups
    all_groups = ["hydro", "wind", "thermal", "solar", "other"]
    current_groups = st.session_state.get("groups", all_groups)

    groups = st.pills(
        "Production groups",
        options=all_groups,
        default=current_groups,
        selection_mode="multi",
    )
    st.session_state["groups"] = groups

st.success(
    f"Stored selection → area: **{st.session_state['price_area']}**, "
    f"year: **{st.session_state['selected_year']}**, "
    f"month: **{st.session_state['month']}**, "
    f"groups: **{', '.join(st.session_state['groups'])}**."
)

st.info(
    "Go to the other pages in the sidebar to view Elhub production, "
    "weather data, STL decomposition, spectrograms, and anomalies "
    "using these default selections."
)
