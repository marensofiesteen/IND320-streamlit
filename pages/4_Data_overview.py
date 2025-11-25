import streamlit as st
import pandas as pd
from pymongo import MongoClient, errors as mongo_errors

from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS
from utils.navigation import sidebar_navigation

# Local meny for this group:
sidebar_navigation("Explorative / overview")

st.set_page_config(page_title="Data overview", layout="wide")

# ---------------------------------------------------------
#  Cached loader for Elhub production_data (MongoDB)
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_production_data() -> pd.DataFrame:
    """
    Load the Elhub production_data collection from MongoDB.

    - reads from the 'production_data' collection
    - normalises column names
    - parses datetime columns
    - derives year and month
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
    if "starttime" in df.columns:
        df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
        df = df.dropna(subset=["starttime"])
        df["year"] = df["starttime"].dt.year
        df["month"] = df["starttime"].dt.month

    if "quantitykwh" in df.columns:
        df = df.dropna(subset=["quantitykwh"])

    return df


# ---------------------------------------------------------
#  Cached loader for Open-Meteo data
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def get_openmeteo_hourly(area_code: str, year: int) -> pd.DataFrame:
    """
    Cached wrapper around fetch_openmeteo_hourly for Open-Meteo ERA5 data.
    """
    try:
        df = fetch_openmeteo_hourly(area_code, year)
    except Exception as e:
        st.error(f"Error while fetching Open-Meteo data for {area_code}, {year}: {e}")
        return pd.DataFrame()

    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            st.error("Open-Meteo data could not be converted to a DataFrame.")
            return pd.DataFrame()

    return df


# ---------------------------------------------------------
#  Elhub Production (MongoDB)
# ---------------------------------------------------------
st.header("Elhub production (from MongoDB)")

with st.spinner("Loading Elhub production data ..."):
    df_prod = load_production_data()

if df_prod.empty:
    st.warning("No production data found in MongoDB. Elhub overview is not available.")
else:
    elhub_areas = sorted(df_prod["pricearea"].dropna().unique())

    # Defaults from session_state
    default_area = st.session_state.get("price_area", "NO1")
    if default_area not in elhub_areas:
        default_area = elhub_areas[0]

    default_month = int(st.session_state.get("month", 1))
    if default_month < 1 or default_month > 12:
        default_month = 1

    # Use selected_year if present, otherwise infer from data
    available_years = sorted(df_prod["year"].unique())
    default_year = int(st.session_state.get("selected_year", available_years[0]))
    if default_year not in available_years:
        default_year = available_years[0]

    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        pa = st.selectbox(
            "Price area (Elhub)",
            elhub_areas,
            index=elhub_areas.index(default_area),
        )
        st.session_state["price_area"] = pa

    with col_e2:
        yr = st.selectbox(
            "Year",
            available_years,
            index=available_years.index(default_year),
        )
        st.session_state["selected_year"] = int(yr)

    with col_e3:
        mo = st.select_slider(
            "Month",
            options=list(range(1, 13)),
            value=default_month,
        )
        st.session_state["month"] = int(mo)

    # Filter for selected area/year/month
    mask = (
        (df_prod["pricearea"] == pa)
        & (df_prod["year"] == yr)
        & (df_prod["month"] == mo)
    )

    df_elhub = df_prod.loc[mask].copy()

    if df_elhub.empty:
        st.warning("No Elhub production data found for the selected area/year/month.")
    else:
        st.dataframe(df_elhub.head(200), use_container_width=True)
        st.caption(
            f"Rows (selected area/year/month: {pa}, {yr}, {mo:02d}): {len(df_elhub):,}"
        )


# ---------------------------------------------------------
#  Open-Meteo Hourly (API, 2021)
# ---------------------------------------------------------
st.header("Open-Meteo hourly (API, 2021)")

# PRICE_AREAS can be a dict or list; we use its keys if dict
if isinstance(PRICE_AREAS, dict):
    weather_areas = list(PRICE_AREAS.keys())
else:
    weather_areas = list(PRICE_AREAS)

if not weather_areas:
    st.error("PRICE_AREAS is empty or not configured. Cannot select weather area.")
else:
    default_weather_area = st.session_state.get("price_area", "NO1")
    if default_weather_area not in weather_areas:
        default_weather_area = weather_areas[0]

    pa2 = st.selectbox(
        "Price area (weather)",
        weather_areas,
        index=weather_areas.index(default_weather_area),
    )
    # Keep global price area in sync (using the latest choice)
    st.session_state["price_area"] = pa2

    with st.spinner("Loading Open-Meteo ERA5 hourly data (2021) ..."):
        df_om = get_openmeteo_hourly(pa2, 2021)

    if df_om.empty:
        st.warning("No data from Open-Meteo (check API and parameters).")
    else:
        st.dataframe(df_om.head(200), use_container_width=True)
        st.caption(f"Rows (full year): {len(df_om):,}")
