import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
from pymongo import MongoClient, errors as mongo_errors
import requests

from utils.navigation import sidebar_navigation

# Local meny for this group:
sidebar_navigation("Explorative / overview")

st.set_page_config(page_title="Meteorology vs Energy – Sliding Window Correlation",
                   layout="wide")
st.title("Meteorology vs Energy – Sliding Window Correlation")

# ---------------------------------------------------------
#  Data loaders
# ---------------------------------------------------------


@st.cache_data(show_spinner=True)
def load_elhub_collection(collection_name: str) -> pd.DataFrame:
    """
    Load a MongoDB collection (production_data or consumption_data) into a DataFrame.
    Uses connection details from Streamlit secrets.
    Includes simple error handling for connection issues.
    """
    try:
        mongo_secrets = st.secrets["mongo"]
        uri = mongo_secrets["uri"]
        db_name = mongo_secrets["db"]
    except Exception as e:
        st.error(f"Missing or invalid MongoDB secrets: {e}")
        return pd.DataFrame()

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Trigger connection test
        client.server_info()
        db = client[db_name]
        coll = db[collection_name]

        docs = list(coll.find({}, {"_id": 0}))
    except mongo_errors.ServerSelectionTimeoutError:
        st.error("Could not reach MongoDB server. Check your network or MongoDB settings.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error while loading data from MongoDB: {e}")
        return pd.DataFrame()

    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df.columns = [c.lower() for c in df.columns]

    # Parse datetime column
    if "starttime" in df.columns:
        df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
        df = df.dropna(subset=["starttime"])

    return df


# Prisområde -> representativ by + koordinater
PRICE_AREA_LOCATIONS = {
    "NO1": {"city": "Oslo",         "lat": 59.9139,  "lon": 10.7522},
    "NO2": {"city": "Kristiansand", "lat": 58.1467,  "lon": 7.9956},
    "NO3": {"city": "Trondheim",    "lat": 63.4305,  "lon": 10.3951},
    "NO4": {"city": "Tromsø",       "lat": 69.6492,  "lon": 18.9553},
    "NO5": {"city": "Bergen",       "lat": 60.39299, "lon": 5.32415},
}


@st.cache_data(show_spinner=True)
def load_weather_from_open_meteo(price_area: str,
                                 start_time: pd.Timestamp,
                                 end_time: pd.Timestamp) -> pd.DataFrame:
    """
    Hent værdata direkte fra Open-Meteo ERA5 for en prisområde-representativ by
    og en gitt tidsperiode (brukes til sliding window correlation).

    Har enkel error handling dersom API-kallet feiler.
    """
    if price_area not in PRICE_AREA_LOCATIONS:
        st.error(f"Unknown price area '{price_area}' for weather download.")
        return pd.DataFrame()

    loc = PRICE_AREA_LOCATIONS[price_area]
    lat = loc["lat"]
    lon = loc["lon"]

    base_url = "https://archive-api.open-meteo.com/v1/era5"

    # Rens start/slutt til datoer
    start_date = start_time.floor("D")
    end_date = end_time.ceil("D")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
        ],
        "timezone": "Europe/Oslo",
    }

    try:
        resp = requests.get(base_url, params=params, timeout=60)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach Open-Meteo API: {e}")
        return pd.DataFrame()

    if resp.status_code != 200:
        st.error(
            f"Failed to fetch weather data from Open-Meteo "
            f"(status {resp.status_code})."
        )
        return pd.DataFrame()

    try:
        data = resp.json()
    except ValueError as e:
        st.error(f"Could not decode Open-Meteo response as JSON: {e}")
        return pd.DataFrame()

    if "hourly" not in data or "time" not in data["hourly"]:
        st.error("Unexpected response format from Open-Meteo (missing 'hourly/time').")
        return pd.DataFrame()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.rename(columns={"time": "date"})

    # Enkle, rene navn (lowercase)
    clean_cols = {}
    for c in df.columns:
        if c == "date":
            clean_cols[c] = "date"
        else:
            clean_cols[c] = c.strip().lower()
    df = df.rename(columns=clean_cols)

    df = df.sort_values("date")
    df["year"] = df["date"].dt.year
    return df


# ---------------------------------------------------------
#  Sliding window correlation helper
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_sliding_correlation(
    df_weather: pd.DataFrame,
    df_energy: pd.DataFrame,
    weather_col: str,
    window_hours: int,
    lag_hours: int,
) -> pd.DataFrame:
    """
    Merge weather and energy on a common hourly timeline and compute
    sliding window correlation between one weather column and energy quantity.
    """
    dfw = df_weather[["date", weather_col]].rename(columns={"date": "time"}).copy()
    dfe = df_energy[["starttime", "quantitykwh"]].rename(columns={"starttime": "time"}).copy()

    # Resample to hourly (vær: mean, energi: sum)
    dfw = dfw.set_index("time").resample("1H").mean()
    dfe = dfe.set_index("time").resample("1H").sum()

    common_index = dfw.index.intersection(dfe.index)
    dfw = dfw.loc[common_index]
    dfe = dfe.loc[common_index]

    if dfw.empty or dfe.empty:
        return pd.DataFrame()

    merged = pd.concat([dfw, dfe], axis=1)
    merged = merged.dropna(subset=[weather_col, "quantitykwh"])

    # Lag
    if lag_hours != 0:
        merged["quantitykwh_shifted"] = merged["quantitykwh"].shift(periods=lag_hours)
    else:
        merged["quantitykwh_shifted"] = merged["quantitykwh"]

    merged = merged.dropna(subset=[weather_col, "quantitykwh_shifted"])

    if merged.empty:
        return pd.DataFrame()

    merged["corr"] = (
        merged[weather_col]
        .rolling(window=window_hours)
        .corr(merged["quantitykwh_shifted"])
    )

    merged = merged.reset_index().rename(columns={"index": "time"})
    return merged


# ---------------------------------------------------------
#  Load energy data first (we need period & years)
# ---------------------------------------------------------


with st.spinner("Loading energy data from MongoDB ..."):
    df_prod = load_elhub_collection("production_data")
    df_cons = load_elhub_collection("consumption_data")

if df_prod.empty and df_cons.empty:
    st.error("No energy data available from MongoDB (both production and consumption are empty).")
    st.stop()

if df_prod.empty:
    st.warning("Production data not available.")
if df_cons.empty:
    st.warning("Consumption data not available.")

# ---------------------------------------------------------
#  Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("Data selection")

data_type = st.sidebar.radio(
    "Energy data type",
    ["Production", "Consumption"],
    index=0,
    horizontal=True,
)

if data_type == "Production":
    if df_prod.empty:
        st.sidebar.error("Production data not available.")
        st.stop()
    df_energy_all = df_prod.copy()
    group_col = "productiongroup"
    title_prefix = "Production"
else:
    if df_cons.empty:
        st.sidebar.error("Consumption data not available.")
        st.stop()
    df_energy_all = df_cons.copy()
    group_col = "consumptiongroup"
    title_prefix = "Consumption"

required_cols_energy = {"pricearea", group_col, "starttime", "quantitykwh"}
missing_energy = required_cols_energy - set(df_energy_all.columns)
if missing_energy:
    st.error(f"Missing expected energy columns: {missing_energy}")
    st.stop()

# Legg til år
df_energy_all["year"] = df_energy_all["starttime"].dt.year

available_years = sorted(df_energy_all["year"].unique())
year = st.sidebar.selectbox("Year", available_years)

available_areas = sorted(df_energy_all["pricearea"].dropna().unique())
price_area = st.sidebar.selectbox("Price area", available_areas)

available_groups = sorted(df_energy_all[group_col].dropna().unique())
group = st.sidebar.selectbox("Energy group", available_groups)

# Korrelasjonsparametre
st.sidebar.header("Correlation parameters")
window_days = st.sidebar.slider("Window size (days)", min_value=3, max_value=90, value=30, step=1)
window_hours = window_days * 24

lag_hours = st.sidebar.slider(
    "Lag (hours)",
    min_value=-72,
    max_value=72,
    value=0,
    step=1,
    help="Positive: energy is shifted forward in time relative to weather.",
)

# ---------------------------------------------------------
#  Filter energy data based on selections
# ---------------------------------------------------------

mask_energy = (
    (df_energy_all["pricearea"] == price_area)
    & (df_energy_all[group_col] == group)
    & (df_energy_all["year"] == year)
)

df_energy_sel = df_energy_all.loc[mask_energy].copy()

if df_energy_sel.empty:
    st.warning("No energy data for the selected combination of area/group/year.")
    st.stop()

energy_start = df_energy_sel["starttime"].min()
energy_end = df_energy_sel["starttime"].max()

# ---------------------------------------------------------
#  Load matching weather data from Open-Meteo
# ---------------------------------------------------------

with st.spinner("Downloading weather data from Open-Meteo ..."):
    df_weather = load_weather_from_open_meteo(
        price_area=price_area,
        start_time=energy_start,
        end_time=energy_end,
    )

if df_weather.empty:
    st.error("No weather data returned for this period/price area.")
    st.stop()

# Velg bare aktuelle året i vær (for sikkerhets skyld)
df_weather_sel = df_weather[df_weather["year"] == year].copy()
if df_weather_sel.empty:
    st.warning("No weather data for the selected year after filtering.")
    st.stop()

# Mulige værvariabler
weather_columns = [
    c for c in df_weather_sel.columns
    if c not in ["date", "year"] and np.issubdtype(df_weather_sel[c].dtype, np.number)
]
if not weather_columns:
    st.error("No numeric weather columns found in weather data.")
    st.stop()

weather_col = st.sidebar.selectbox("Weather variable", weather_columns)

# ---------------------------------------------------------
#  Compute sliding correlation
# ---------------------------------------------------------

with st.spinner("Computing sliding window correlation ..."):
    df_corr = compute_sliding_correlation(
        df_weather=df_weather_sel,
        df_energy=df_energy_sel,
        weather_col=weather_col,
        window_hours=window_hours,
        lag_hours=lag_hours,
    )

if df_corr.empty:
    st.warning("No overlapping data after merging and lagging. Try a different year, lag, or window.")
    st.stop()

# ---------------------------------------------------------
#  Layout: time series + correlation plots
# ---------------------------------------------------------

ts_col, corr_col = st.columns(2)

location_info = PRICE_AREA_LOCATIONS.get(price_area, None)
city_name = location_info["city"] if location_info else price_area

with ts_col:
    st.subheader("Time series – weather and energy")
    st.caption(
        f"{title_prefix} vs {weather_col} for {city_name} "
        f"({price_area}), group {group}, year {year}"
    )

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])

    # Energi (venstre y-akse)
    fig_ts.add_trace(
        {
            "x": df_corr["time"],
            "y": df_corr["quantitykwh_shifted"],
            "name": "quantitykwh_shifted",
        },
        secondary_y=False,
    )

    # Vær (høyre y-akse)
    fig_ts.add_trace(
        {
            "x": df_corr["time"],
            "y": df_corr[weather_col],
            "name": weather_col,
        },
        secondary_y=True,
    )

    fig_ts.update_layout(
        title="Weather vs shifted energy time series",
        hovermode="x unified",
        legend_title_text="Series",
    )

    fig_ts.update_xaxes(title_text="Time")
    fig_ts.update_yaxes(title_text="Energy (kWh)", secondary_y=False)
    fig_ts.update_yaxes(title_text=weather_col, secondary_y=True)

    st.plotly_chart(fig_ts, use_container_width=True)

with corr_col:
    st.subheader("Sliding window correlation")
    st.caption(
        f"Rolling correlation over {window_days} days, "
        f"lag = {lag_hours} hours"
    )

    fig_corr = px.line(
        df_corr,
        x="time",
        y="corr",
        labels={"corr": "Correlation", "time": "Time"},
        title="Sliding window correlation",
    )
    fig_corr.add_hline(y=0, line_dash="dash")
    fig_corr.update_yaxes(range=[-1, 1])
    fig_corr.update_layout(hovermode="x unified")
    st.plotly_chart(fig_corr, use_container_width=True)

with st.expander("How to interpret this page"):
    st.markdown(
        f"""
        - This page compares **weather** (`{weather_col}`) with **{title_prefix.lower()}**
          (`quantitykwh`) for:
          - Price area **{price_area}** (represented by **{city_name}**)
          - Group **{group}**
          - Year **{year}**
        - Weather data are downloaded on-the-fly from **Open-Meteo ERA5**.
        - The energy series is optionally **lagged** by *lag_hours* before merging.
        - A sliding window of **{window_days} days** is used to compute the rolling
          Pearson correlation between the selected weather variable and the (lagged)
          energy series.
        - Error handling:
          - If MongoDB or Open-Meteo are unavailable, the page shows a clear error
            instead of crashing.
          - If data filters result in empty datasets, informative warnings are shown.
        """
    )
