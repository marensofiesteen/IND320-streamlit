import streamlit as st
import pandas as pd

from analysis_utils import satv_spc_plot, lof_anomaly_plot
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS
from utils.navigation import sidebar_navigation

# Local meny for this group
sidebar_navigation("Anomalies / data quality")

st.set_page_config(page_title="Outliers & LOF", layout="wide")
st.title("Outlier/SPC (SATV) and LOF (Open-Meteo)")

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
#  Global price area selection for weather data
# ---------------------------------------------------------

# PRICE_AREAS can be a dict or a list-like
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
    "Price area (weather)",
    weather_areas,
    index=weather_areas.index(default_area),
)

# Keep global selection in sync so other pages see the same area
st.session_state["price_area"] = pa

# ---------------------------------------------------------
#  Load weather data
# ---------------------------------------------------------
with st.spinner("Loading Open-Meteo ERA5 hourly data (2021) ..."):
    df = get_openmeteo_hourly(pa, 2021)

if df.empty:
    st.error("No Open-Meteo data for the selected area/year.")
    st.stop()

# Basic checks that required variables exist
if "temperature_2m" not in df.columns:
    st.error("Column 'temperature_2m' is missing in the weather data.")
    st.stop()

if "precipitation" not in df.columns:
    st.error("Column 'precipitation' is missing in the weather data.")
    st.stop()


tab1, tab2 = st.tabs(
    [
        "SATV + SPC (temperature)",
        "LOF anomalies (precipitation)",
    ]
)

# ----------------- TAB 1: SATV + SPC -----------------
with tab1:
    st.subheader("SATV-based SPC on temperature")

    k = st.slider(
        "SPC σ multiplier",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Number of robust standard deviations used for SPC limits.",
    )

    cutoff = st.number_input(
        "DCT cutoff (low-frequency components kept in trend)",
        min_value=8,
        value=200,
        step=4,
        help="Controls how smooth the trend is. Larger cutoff ⇒ more detail in the trend.",
    )

    with st.spinner("Computing SATV and SPC boundaries..."):
        try:
            fig_satv, out_satv = satv_spc_plot(
                df,
                temp_col="temperature_2m",
                cutoff=int(cutoff),
                k_sigma=float(k),
            )
            st.plotly_chart(fig_satv, use_container_width=True)

            if isinstance(out_satv, pd.DataFrame) and not out_satv.empty:
                st.dataframe(out_satv.head(20), use_container_width=True)
                st.caption(f"Number of outliers (SATV/SPC): {len(out_satv):,}")
            else:
                st.info("No temperature outliers detected with the current parameters.")
        except Exception as e:
            st.error(f"SATV/SPC error: {e}")

# ----------------- TAB 2: LOF anomalies -----------------
with tab2:
    st.subheader("LOF anomalies on precipitation")

    cont = st.slider(
        "LOF contamination (proportion of anomalies)",
        min_value=0.001,
        max_value=0.05,
        value=0.01,
        step=0.001,
        help="Approximate proportion of points to flag as anomalies.",
    )

    nn = st.number_input(
        "LOF n_neighbors",
        min_value=5,
        value=20,
        step=1,
        help="Number of neighbors used in the Local Outlier Factor algorithm.",
    )

    with st.spinner("Computing LOF anomalies on precipitation..."):
        try:
            fig_lof, out_lof = lof_anomaly_plot(
                df,
                var_col="precipitation",
                contamination=float(cont),
                n_neighbors=int(nn),
            )
            st.plotly_chart(fig_lof, use_container_width=True)

            if isinstance(out_lof, pd.DataFrame) and not out_lof.empty:
                st.dataframe(out_lof.head(20), use_container_width=True)
                st.caption(f"Number of LOF anomalies: {len(out_lof):,}")
            else:
                st.info("No LOF anomalies detected with the current parameters.")
        except Exception as e:
            st.error(f"LOF error: {e}")
