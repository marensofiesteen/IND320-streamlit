import streamlit as st

from analysis_utils import satv_spc_plot, lof_anomaly_plot
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS

st.set_page_config(page_title="Outliers & LOF", layout="wide")
st.title("Outlier/SPC (SATV) and LOF (Open-Meteo)")

# --- Global price area selection for weather data ---
weather_areas = list(PRICE_AREAS.keys())

default_area = st.session_state.get("price_area", "NO1")
if default_area not in weather_areas:
    default_area = "NO1"

pa = st.selectbox(
    "Price area (weather)",
    weather_areas,
    index=weather_areas.index(default_area),
)

# Keep global selection in sync so other pages see the same area
st.session_state["price_area"] = pa

df = fetch_openmeteo_hourly(pa, 2021)
if df.empty:
    st.error("No Open-Meteo data.")
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
        help="Controls how smooth the trend is. Larger cutoff => more detail in the trend.",
    )

    fig_satv, out_satv = satv_spc_plot(
        df,
        temp_col="temperature_2m",
        cutoff=int(cutoff),
        k_sigma=float(k),
    )
    st.plotly_chart(fig_satv, use_container_width=True)
    st.dataframe(out_satv.head(20), use_container_width=True)

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

    fig_lof, out_lof = lof_anomaly_plot(
        df,
        var_col="precipitation",
        contamination=float(cont),
        n_neighbors=int(nn),
    )
    st.plotly_chart(fig_lof, use_container_width=True)
    st.dataframe(out_lof.head(20), use_container_width=True)

