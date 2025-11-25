import streamlit as st

from utils.navigation import sidebar_navigation

sidebar_navigation("Explorative / overview")

st.title("Energy, Meteorology & Production Dashboard")
st.caption("Global selection page – default values shared across the entire app")

st.markdown(
    """
    Welcome! This app is structured into four groups:

    1. **Explorative / overview** – overview of energy and weather data  
    2. **Anomalies / data quality** – outliers and strange patterns  
    3. **Forecasting** – SARIMAX-based forecasts of production/consumption  
    4. **Snow & geo** – snow-related analysis and maps
    """
)

# ---------------------------------------------------------
#  Global defaults stored in session_state
# ---------------------------------------------------------

if "price_area" not in st.session_state:
    st.session_state["price_area"] = "NO1"

# Part 3 primarily uses 2021 as the reference year
if "selected_year" not in st.session_state:
    st.session_state["selected_year"] = 2021

if "month" not in st.session_state:
    st.session_state["month"] = 1

if "groups" not in st.session_state:
    st.session_state["groups"] = ["hydro", "wind", "thermal", "solar", "other"]

# ---------------------------------------------------------
#  Introduction
# ---------------------------------------------------------

st.markdown(
    """
This page defines **global default selections** for the entire application:

- Electricity **price area** (NO1–NO5)  
- **Year** (Elhub production reference year, mainly 2021)  
- **Month** (used by pages that work with monthly or seasonal subsets)  
- **Production groups**  

All selections are stored in `st.session_state` and can be accessed
by the other pages (Elhub production, weather, STL decomposition,
spectrogram, anomalies, sliding correlation, snow drift, forecasting, etc.).
"""
)

# ---------------------------------------------------------
#  UI – Global selectors
# ---------------------------------------------------------

areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
years = [2021]  # Project specification: 2021 used consistently in part 3

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
    # Global year and month
    current_year = st.session_state.get("selected_year", 2021)
    year = st.selectbox(
        "Year (default for several pages)",
        years,
        index=years.index(current_year),
    )
    st.session_state["selected_year"] = year

    current_month = st.session_state.get("month", 1)
    month = st.select_slider(
        "Month (1–12)",
        options=list(range(1, 13)),
        value=current_month,
    )
    st.session_state["month"] = month

with col3:
    # Production groups
    all_groups = ["hydro", "wind", "thermal", "solar", "other"]
    current_groups = st.session_state.get("groups", all_groups)

    groups = st.pills(
        "Production groups (default)",
        options=all_groups,
        default=current_groups,
        selection_mode="multi",
    )
    st.session_state["groups"] = groups

# ---------------------------------------------------------
#  Confirmation box
# ---------------------------------------------------------

st.success(
    f"Stored global selection → "
    f"area: **{st.session_state['price_area']}**, "
    f"year: **{st.session_state['selected_year']}**, "
    f"month: **{st.session_state['month']}**, "
    f"groups: **{', '.join(st.session_state['groups'])}**."
)

# ---------------------------------------------------------
#  Guidance for navigating the app
# ---------------------------------------------------------

st.info(
    """
Use the sidebar to navigate to:

- **Elhub production/consumption pages**  
- **Meteorology pages** (ERA5 data from Open-Meteo)  
- **STL decomposition** and **spectrogram**  
- **Outlier detection** (SPC & LOF)  
- **Meteorology vs Energy** (sliding window correlation)  
- **Snow drift & wind rose analysis**  
- **SARIMAX forecasting** with optional exogenous weather variables  

These selections act as *global defaults*—each analysis page may also
provide additional local filters when needed.
"""
)

with st.expander("Technical notes"):
    st.markdown(
        """
        **Technical notes**
        - All selections are stored using `st.session_state` and can be reused across pages.
        - Pages may override these defaults if more detailed local control is required.
        - This structure ensures consistent navigation and reproducible settings.
        """
    )
