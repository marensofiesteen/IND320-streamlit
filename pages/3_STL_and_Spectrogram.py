import streamlit as st
import pandas as pd
from pymongo import MongoClient, errors as mongo_errors

from analysis_utils import stl_decompose_production, spectrogram_production
from utils.utils_openmeteo import PRICE_AREAS
from utils.navigation import sidebar_navigation

# Local meny for this group
sidebar_navigation("Anomalies / data quality")

st.set_page_config(page_title="STL & Spectrogram", layout="wide")
st.title("STL and Spectrogram (Elhub production)")

# ---------------------------------------------------------
#  MongoDB loader for production data
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_production_data() -> pd.DataFrame:
    """
    Load the Elhub production_data collection from MongoDB.

    - reads from the 'production_data' collection
    - normalises column names
    - parses datetime columns
    - drops rows with invalid timestamps or missing quantity
    """
    try:
        mongo_secrets = st.secrets["mongo"]
        uri = mongo_secrets["uri"]
        db_name = mongo_secrets["db"]
    except Exception as e:
        st.error(f"Missing or invalid MongoDB configuration in secrets: {e}")
        return pd.DataFrame()

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[db_name]
        coll = db["production_data"]
        docs = list(coll.find({}, {"_id": 0}))
    except mongo_errors.ServerSelectionTimeoutError:
        st.error("Could not reach MongoDB server. Check network or MongoDB settings.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error while loading production data from MongoDB: {e}")
        return pd.DataFrame()

    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df.columns = [c.lower() for c in df.columns]

    # Parse datetime-like columns
    for col in ["starttime", "endtime", "lastupdatedtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop rows with invalid timestamps or missing quantity
    if "starttime" in df.columns:
        df = df.dropna(subset=["starttime"])
    if "quantitykwh" in df.columns:
        df = df.dropna(subset=["quantitykwh"])

    # Derive year
    if "starttime" in df.columns:
        df["year"] = df["starttime"].dt.year

    return df


# ---------------------------------------------------------
#  Global selection of price area, production group, year
# ---------------------------------------------------------

# Use PRICE_AREAS if defined as a dict or list; fall back to NO1–NO5
if isinstance(PRICE_AREAS, dict):
    ALL_AREAS = sorted(list(PRICE_AREAS.keys()))
else:
    ALL_AREAS = sorted(list(PRICE_AREAS)) if PRICE_AREAS else ["NO1", "NO2", "NO3", "NO4", "NO5"]

default_area = st.session_state.get("price_area", "NO1")
if default_area not in ALL_AREAS:
    default_area = ALL_AREAS[0]

# Load production data first to know which years exist
with st.spinner("Loading Elhub production data for STL/Spectrogram ..."):
    df_prod = load_production_data()

if df_prod.empty:
    st.error("No production data found in MongoDB. STL and spectrogram cannot be computed.")
    st.stop()

# Valid years in data (expected 2021–2024)
available_years = sorted(df_prod["year"].unique()) if "year" in df_prod.columns else []
if not available_years:
    st.error("No valid years found in production data.")
    st.stop()

default_year = st.session_state.get("selected_year", 2021)
if default_year not in available_years:
    default_year = available_years[0]

col_sel1, col_sel2, col_sel3 = st.columns(3)

with col_sel1:
    area = st.selectbox(
        "Price area",
        ALL_AREAS,
        index=ALL_AREAS.index(default_area),
    )

with col_sel2:
    year = st.selectbox(
        "Year",
        available_years,
        index=available_years.index(default_year),
    )

# Use global groups if available, otherwise fall back to full list
groups = ["hydro", "wind", "thermal", "solar", "other"]
session_groups = st.session_state.get("groups", groups)
if not session_groups:
    session_groups = groups

default_group = st.session_state.get("production_group", session_groups[0])
if default_group not in groups:
    default_group = session_groups[0]

with col_sel3:
    group = st.selectbox(
        "Production group",
        groups,
        index=groups.index(default_group),
    )

# Keep global selections in sync so other pages can use them
st.session_state["price_area"] = area
st.session_state["production_group"] = group
st.session_state["selected_year"] = int(year)


# ---------------------------------------------------------
#  Filter production data for selected area/year/group
# ---------------------------------------------------------
if "productiongroup" not in df_prod.columns:
    st.error("Column 'productiongroup' is missing in production_data.")
    st.stop()

mask = (
    (df_prod["pricearea"] == area)
    & (df_prod["productiongroup"] == group)
    & (df_prod["year"] == year)
)

df = df_prod.loc[mask].copy()

if df.empty:
    st.error("No Elhub production data found for the selected area/year/group.")
    st.stop()

if "starttime" not in df.columns:
    st.error("Column 'starttime' is missing in production_data; cannot build time series.")
    st.stop()

# ---------------------------------------------------------
#  Build clean hourly time series with unique timestamps
#  and keep pricearea + productiongroup
# ---------------------------------------------------------
df["datetime"] = df["starttime"]

ts = (
    df[["datetime", "quantitykwh", "pricearea", "productiongroup"]]
    .dropna(subset=["datetime", "quantitykwh"])
    .sort_values("datetime")
    .groupby(
        ["pricearea", "productiongroup", pd.Grouper(key="datetime", freq="1H")],
        as_index=False,
    )["quantitykwh"]
    .sum()
)

tab1, tab2 = st.tabs(["STL", "Spectrogram"])

# ----------------- TAB 1: STL -----------------
with tab1:
    st.subheader("STL decomposition")

    # Slider for seasonal period
    period = st.slider(
        "Seasonal period (hours)",
        min_value=24,
        max_value=24 * 28,   # up to 4 weeks
        value=168,
        step=24,
        help="Typical choice for hourly data is 168 hours (1 week).",
    )

    # Default trend window: at least period + some buffer
    trend_default = max(31 * 24, int(period) + 24)

    trend = st.slider(
        "Trend smoother (hours)",
        min_value=int(period) + 1,
        max_value=24 * 180,   # up to ~6 months of smoothing
        value=trend_default,
        step=24,
        help="Larger values give a smoother trend curve.",
    )

    # Spinner while STL is computed
    with st.spinner("Computing STL decomposition..."):
        try:
            fig_stl, _ = stl_decompose_production(
                ts,
                area,
                group,
                period_length=int(period),
                trend_smoother=int(trend),
            )
            st.plotly_chart(fig_stl, use_container_width=True)
        except Exception as e:
            st.error(f"STL error: {e}")

# ----------------- TAB 2: Spectrogram -----------------
with tab2:
    st.subheader("Spectrogram")

    wl = st.slider(
        "Window length (hours)",
        min_value=32,
        max_value=24 * 60,   # up to ~2 months
        value=168,
        step=8,
        help="Length of the STFT analysis window.",
    )

    ov = st.slider(
        "Window overlap (fraction)",
        min_value=0.0,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Fractional overlap between consecutive windows.",
    )

    with st.spinner("Computing spectrogram..."):
        try:
            fig_spec, _ = spectrogram_production(
                ts,
                area,
                group,
                window_length_hours=int(wl),
                window_overlap=float(ov),
            )
            st.plotly_chart(fig_spec, use_container_width=True)
        except Exception as e:
            st.error(f"Spectrogram error: {e}")

# ---------------------------------------------------------
#  Interpretation help
# ---------------------------------------------------------
with st.expander("How to interpret this page"):
    st.markdown(
        f"""
        - This page uses **STL decomposition** and a **spectrogram** to analyse 
          hourly Elhub production data for:
          - Price area: **{area}**
          - Production group: **{group}**
          - Year: **{year}**
        - **STL decomposition** separates the time series into:
          - trend, seasonal and residual components,
          - with user-controlled seasonal period and trend smoother.
        - The **spectrogram** shows how the frequency content of the production
          time series changes over time (based on a sliding STFT window).
        - Waiting time is handled by:
          - `@st.cache_data` for loading production data from MongoDB,
          - `st.spinner(...)` for longer STL and spectrogram computations.
        - Errors related to missing data or failing database calls are caught
          and reported as friendly messages instead of raw tracebacks.
        """
    )
