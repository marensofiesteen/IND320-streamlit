import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from datetime import datetime

from utils.navigation import sidebar_navigation

# Local meny for this group
sidebar_navigation("Snow & geo")

st.set_page_config(page_title="Snow drift", layout="wide")
st.title("Snow drift and wind rose (hydrological years)")

# ---------- Helpers copied/adapted from the teacher's Snow_drift.py ----------

def compute_Qupot(hourly_wind_speeds, dt=3600):
    """
    Compute potential wind-driven snow transport (Qupot) [kg/m]
    by summing hourly contributions using u^3.8 (Tabler, 2003).
    Qupot = sum((u^3.8) * dt) / 233847
    """
    total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847
    return total


def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    """
    Simplified Tabler-style snow transport.
    Parameters:
        T     : Maximum transport distance (m)
        F     : Fetch distance (m)
        theta : Relocation coefficient (-)
        Swe   : Total snowfall water equivalent (mm)
        hourly_wind_speeds : iterable of wind speeds [m/s]
        dt    : timestep [s]

    Returns a dict with:
        Qupot, Qspot, Srwe, Qinf, Qt, Control
    """
    # 1) Potential wind-driven transport
    Qupot = compute_Qupot(hourly_wind_speeds, dt=dt)

    # 2) Snowfall-limited transport (teacher’s description)
    Qspot = 0.5 * T * Swe  # [kg/m], up to a constant

    # 3) Relocated water equivalent
    Srwe = theta * Swe  # [mm], proportional to Swe

    # 4) Controlling process (simplified):
    #    If snowfall-limited transport is smaller -> snowfall controls
    #    Otherwise wind controls.
    if Qspot < Qupot:
        Qinf = Qspot
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"

    # 5) Total transport over fetch F (Tabler-type factor)
    Qt = Qinf * (1 - 0.14 ** (F / T))

    return {
        "Qupot (kg/m)": Qupot,
        "Qspot (kg/m)": Qspot,
        "Srwe (mm)": Srwe,
        "Qinf (kg/m)": Qinf,
        "Qt (kg/m)": Qt,
        "Control": control,
    }


def sector_index(direction):
    """
    Given a wind direction in degrees, returns the index (0-15)
    corresponding to a 16-sector division (22.5° per sector).
    """
    return int(((direction + 11.25) % 360) // 22.5)


def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    """
    Compute cumulative transport for each of 16 wind sectors.
    Uses the same u^3.8 formulation as Qupot.
    """
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors


# ---------- Open-Meteo: fetch hourly data for one hydrological year ----------

@st.cache_data(show_spinner=True)
def fetch_openmeteo_hourly(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch hourly historical meteorological data from Open-Meteo ERA5 archive.
    Variables:
      - temperature_2m
      - precipitation
      - wind_speed_10m
      - wind_direction_10m
    Returns a DataFrame with a 'time' column in datetime format.

    We use the archive endpoint because we need past years (e.g. 2019–2023).
    """
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m",
        "timezone": "Europe/Oslo",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            st.warning(
                f"Open-Meteo request failed for {start.date()}–{end.date()} "
                f"(status {resp.status_code})."
            )
            return pd.DataFrame()
        data = resp.json()
    except Exception as e:
        st.warning(f"Error contacting Open-Meteo: {e}")
        return pd.DataFrame()

    if "hourly" not in data:
        st.warning("Open-Meteo response did not contain 'hourly' data.")
        return pd.DataFrame()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(show_spinner=False)
def compute_snow_drift_for_hydro_year(
    lat: float,
    lon: float,
    hydro_year: int,
    T: float,
    F: float,
    theta: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute snow drift for one hydrological year:
    1 July hydro_year -> 30 June hydro_year+1.

    Returns:
      df_hourly : hourly dataframe with extra columns (Swe, Swe_hourly)
      result    : dict from compute_snow_transport (Qt, etc.)
    """
    start = datetime(hydro_year, 7, 1, 0, 0)
    end = datetime(hydro_year + 1, 6, 30, 23, 0)

    df = fetch_openmeteo_hourly(lat, lon, start, end)
    if df.empty:
        return df, {}

    df = df.copy()

    # Compute SWE per hour: precipitation when T < 1°C
    df["Swe_hourly"] = np.where(df["temperature_2m"] < 1.0, df["precipitation"], 0.0)
    total_Swe = df["Swe_hourly"].sum()

    wind_speeds = df["wind_speed_10m"].to_numpy()

    result = compute_snow_transport(T, F, theta, total_Swe, wind_speeds)
    result["hydro_year"] = hydro_year

    return df, result


def compute_monthly_snow_drift(
    df_hourly: pd.DataFrame,
    hydro_year: int,
    T: float,
    F: float,
    theta: float,
) -> pd.DataFrame:
    """
    Compute monthly snow drift (Qt) within a hydrological year DataFrame.

    Returns a DataFrame with:
      - hydro_year
      - month_start (timestamp)
      - Qt (kg/m)
      - Qupot, Qspot, Qinf, Srwe, Control
    """
    df = df_hourly.copy()
    if "time" not in df.columns or "Swe_hourly" not in df.columns:
        return pd.DataFrame()

    df["month_start"] = df["time"].dt.to_period("M").dt.to_timestamp()

    results = []
    for month_val, df_m in df.groupby("month_start"):
        if df_m.empty:
            continue

        Swe_m = df_m["Swe_hourly"].sum()
        ws_m = df_m["wind_speed_10m"].to_numpy()
        res = compute_snow_transport(T, F, theta, Swe_m, ws_m)
        res["hydro_year"] = hydro_year
        res["month_start"] = month_val
        results.append(res)

    if not results:
        return pd.DataFrame()

    monthly_df = pd.DataFrame(results)
    return monthly_df


def aggregate_snow_drift(
    lat: float,
    lon: float,
    year_start: int,
    year_end: int,
    T: float,
    F: float,
    theta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """
    Loop over hydrological years and aggregate Qt per year and per month.

    Returns:
      yearly_df  : DataFrame with one row per hydrological year
      monthly_df : DataFrame with one row per month in the selected hydrological years
      dfs_hourly : list of hourly DataFrames for wind-rose construction
    """
    results_yearly = []
    results_monthly = []
    hourly_list = []

    year_range = list(range(year_start, year_end + 1))
    n_years = len(year_range)

    progress = st.progress(0.0, text="Downloading and aggregating snow drift data ...")

    for idx, y in enumerate(year_range):
        progress.progress(
            (idx + 1) / n_years,
            text=f"Hydrological year {y} (1 July {y} – 30 June {y+1})",
        )

        df_y, res_y = compute_snow_drift_for_hydro_year(lat, lon, y, T, F, theta)
        if df_y.empty or not res_y:
            continue

        hourly_list.append(df_y)
        results_yearly.append(res_y)

        monthly_df_y = compute_monthly_snow_drift(df_y, y, T, F, theta)
        if not monthly_df_y.empty:
            results_monthly.append(monthly_df_y)

    progress.empty()

    if not results_yearly:
        return pd.DataFrame(), pd.DataFrame(), []

    yearly_df = pd.DataFrame(results_yearly)

    if results_monthly:
        monthly_df = pd.concat(results_monthly, ignore_index=True)
    else:
        monthly_df = pd.DataFrame()

    return yearly_df, monthly_df, hourly_list


def build_wind_rose_from_hourly(dfs_hourly: list[pd.DataFrame]):
    """
    Build a simple wind rose from a list of hourly dataframes.
    Uses mean sector-wise transport (kg/m) and plots tonnes/m as radius.
    """
    if not dfs_hourly:
        return None

    sector_values_list = []
    for df in dfs_hourly:
        df = df.dropna(subset=["wind_speed_10m", "wind_direction_10m"]).copy()
        if df.empty:
            continue
        ws = df["wind_speed_10m"].to_numpy()
        wdir = df["wind_direction_10m"].to_numpy()
        sectors = compute_sector_transport(ws, wdir)
        sector_values_list.append(sectors)

    if not sector_values_list:
        return None

    avg_sectors = np.mean(sector_values_list, axis=0)  # kg/m
    sector_tonnes = avg_sectors / 1000.0

    sector_labels = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW",
    ]

    df_rose = pd.DataFrame({
        "direction": sector_labels,
        "Qt_tonnes_per_m": sector_tonnes,
    })

    fig = px.bar_polar(
        df_rose,
        r="Qt_tonnes_per_m",
        theta="direction",
        color="Qt_tonnes_per_m",
        color_continuous_scale="Viridis",
        title="Wind rose – mean transport per direction sector (tonnes/m)",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ---------- Streamlit UI ----------

# 1) Get coordinate from map page
coord = st.session_state.get("selected_coord")
if coord is None:
    st.warning(
        "No coordinate selected yet. "
        "Please go to the map page and click a location first."
    )
    st.stop()

lat = coord["lat"]
lon = coord["lon"]
st.info(f"Using coordinate from map page: **lat = {lat:.4f}, lon = {lon:.4f}**")

# 2) Sidebar: hydrological year range and Tabler parameters
st.sidebar.header("Hydrological year range")

year_start = st.sidebar.number_input(
    "Start hydrological year", min_value=2000, max_value=2030, value=2019, step=1
)
year_end = st.sidebar.number_input(
    "End hydrological year", min_value=2000, max_value=2030, value=2023, step=1
)

if year_start > year_end:
    st.sidebar.error("Start year must be <= end year.")
    st.stop()

max_year_span = 10
if (year_end - year_start + 1) > max_year_span:
    st.sidebar.error(f"Please select a maximum span of {max_year_span} hydrological years.")
    st.stop()

st.sidebar.header("Tabler parameters")

T = st.sidebar.slider(
    "T – maximum transport distance [m]",
    min_value=500.0,
    max_value=5000.0,
    value=3000.0,
    step=250.0,
)
F = st.sidebar.slider(
    "F – fetch distance [m]",
    min_value=5000.0,
    max_value=50000.0,
    value=30000.0,
    step=2500.0,
)
theta = st.sidebar.slider(
    "θ – relocation coefficient [-]",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

# 3) Compute snow drift for selected interval
with st.spinner("Calculating snow drift index for selected years ..."):
    yearly_df, monthly_df, hourly_list = aggregate_snow_drift(
        lat, lon, year_start, year_end, T, F, theta
    )

if yearly_df.empty:
    st.error("No snow drift data could be calculated for the selected period.")
    st.stop()

# 4) Plot yearly + monthly Qt together
left, right = st.columns(2)

with left:
    st.subheader("Snow drift per hydrological year and month")
    st.caption("Hydrological year: 1 July Y – 30 June Y+1")

    # Combined figure: yearly Qt (row 1) and monthly Qt (row 2)
    fig_combo = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15,
        subplot_titles=("Yearly snow transport Qt", "Monthly snow transport Qt"),
    )

    fig_combo.add_trace(
        go.Bar(
            x=yearly_df["hydro_year"],
            y=yearly_df["Qt (kg/m)"],
            name="Yearly Qt [kg/m]",
        ),
        row=1,
        col=1,
    )

    if not monthly_df.empty and "month_start" in monthly_df.columns:
        fig_combo.add_trace(
            go.Bar(
                x=monthly_df["month_start"],
                y=monthly_df["Qt (kg/m)"],
                name="Monthly Qt [kg/m]",
            ),
            row=2,
            col=1,
        )

    fig_combo.update_xaxes(title_text="Hydrological year", row=1, col=1)
    fig_combo.update_xaxes(title_text="Month", row=2, col=1)
    fig_combo.update_yaxes(title_text="Qt [kg/m]", row=1, col=1)
    fig_combo.update_yaxes(title_text="Qt [kg/m]", row=2, col=1)

    fig_combo.update_layout(
        hovermode="x unified",
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig_combo, use_container_width=True)

    st.dataframe(
        yearly_df[
            [
                "hydro_year",
                "Qupot (kg/m)",
                "Qspot (kg/m)",
                "Qinf (kg/m)",
                "Qt (kg/m)",
                "Control",
            ]
        ],
        use_container_width=True,
    )

# 5) Wind rose for the same period
with right:
    st.subheader("Wind rose (Tabler-based sectors)")
    with st.spinner("Building wind rose from hourly data ..."):
        fig_rose = build_wind_rose_from_hourly(hourly_list)

    if fig_rose is None:
        st.info("Not enough wind data to build a wind rose.")
    else:
        st.plotly_chart(fig_rose, use_container_width=True)

with st.expander("About this page"):
    st.markdown(
        """
        - This page adapts the course-provided **Snow_drift.py** script to a Streamlit page.
        - A *hydrological year* is defined as **1 July Y – 30 June Y+1**.
        - For each hydrological year:
          - Hourly data is downloaded from **open-meteo.com**:
            `temperature_2m`, `precipitation`, `wind_speed_10m`, `wind_direction_10m`.
          - Hourly SWE is defined as precipitation when temperature < 1°C.
          - Total SWE and hourly wind speeds are used in a simplified
            **Tabler (2003)**-style snow transport calculation:
            potential (Qupot), snowfall-limited (Qspot), controlling transport (Qinf)
            and total transport Qt over the fetch distance F.
        - The top plot shows **yearly Qt** (kg/m) per hydrological year.
        - The bottom plot shows **monthly Qt** (kg/m) for the same period,
          so you can compare seasonal variation with the yearly totals.
        - A 16-sector wind rose is created using the u^3.8 formulation per direction sector,
          aggregated over all selected hydrological years.
        - The coordinate is taken from the map page (`selected_coord` in `session_state`),
          so you can explore different locations by first clicking on the map.
        - Waiting time is handled by:
          - `@st.cache_data` for ERA5 downloads and per-year snow drift calculations,
          - a progress bar for the hydrological-year loop,
          - `st.spinner(...)` for long-running computations.
        - Errors related to API calls or missing data are caught and reported as
          friendly messages instead of raw tracebacks.
        """
    )
