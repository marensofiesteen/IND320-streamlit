import streamlit as st
import pandas as pd
import plotly.express as px

from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS
from utils.navigation import sidebar_navigation

# Local meny for this group:
sidebar_navigation("Explorative / overview")

st.set_page_config(page_title="Weather plots", layout="wide")
st.header("Time series exploration of weather data (Open-Meteo API, 2021)")

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

    # Ensure we have a datetime-like 'date' column if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    return df


# ---------------------------------------------------------
#  Global price area selection
# ---------------------------------------------------------

# PRICE_AREAS can be a dict or list
if isinstance(PRICE_AREAS, dict):
    weather_areas = list(PRICE_AREAS.keys())
else:
    weather_areas = list(PRICE_AREAS)

if not weather_areas:
    st.error("PRICE_AREAS is empty or not configured. Cannot select weather area.")
    st.stop()

default_area = st.session_state.get("price_area", "NO1")
if default_area not in weather_areas:
    default_area = weather_areas[0]

pa = st.selectbox(
    "Price area",
    weather_areas,
    index=weather_areas.index(default_area),
)

# Keep global selection in sync so other pages see the same area
st.session_state["price_area"] = pa

# ---------------------------------------------------------
#  Load weather data
# ---------------------------------------------------------
with st.spinner("Loading Open-Meteo ERA5 hourly weather data (2021) ..."):
    df = get_openmeteo_hourly(pa, 2021)

if df.empty:
    st.error("No data from Open-Meteo for the selected area/year.")
    st.stop()

if "date" not in df.columns:
    st.error("Weather DataFrame does not contain a 'date' column – cannot make time series plots.")
    st.stop()

# Add year-month label for filtering
df["year_month"] = df["date"].dt.strftime("%Y-%m")

# Build list of possible variables to plot
# Exclude metadata / helper columns
exclude_cols = {"date", "year_month", "pricearea", "city"}
# Normalise candidate list by checking lowercased column names against lowercased exclude set
exclude_lower = {c.lower() for c in exclude_cols}
meteo_options = [
    c for c in df.columns
    if c.lower() not in exclude_lower and pd.api.types.is_numeric_dtype(df[c])
]

if not meteo_options:
    st.error("No numeric meteorological variables found to plot.")
    st.stop()

option_meteo = st.selectbox(
    "Step 1: Select one variable (or 'Show all')",
    options=meteo_options + ["Show all"],
    index=0,
)

# Select month (cumulative until selected month)
month_list = sorted(df["year_month"].dropna().unique().tolist())
if not month_list:
    st.error("No valid year-month values found in weather data.")
    st.stop()

option_month = st.select_slider(
    "Step 2: Select a month (plots all data up to this month)",
    options=month_list,
    value=month_list[0],
)

df_filtered = df[df["year_month"] <= option_month].copy()
if df_filtered.empty:
    st.warning("No data in the selected month range.")
    st.stop()

start_month = df_filtered["year_month"].min()

# ---------------------------------------------------------
#  Plot
# ---------------------------------------------------------
if option_meteo != "Show all":
    fig = px.line(
        df_filtered,
        x="date",
        y=option_meteo,
        labels={"date": "Time", option_meteo: option_meteo},
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
        labels={"date": "Time", "value": "Value", "variable": "Variable"},
        title=f"All variables from {start_month} to {option_month}",
    )

fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
