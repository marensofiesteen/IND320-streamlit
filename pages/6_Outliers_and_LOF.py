import streamlit as st
from analysis_utils import satv_spc_plot, lof_anomaly_plot
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS

st.set_page_config(page_title="Outliers & LOF", layout="wide")
st.title("Outlier/SPC (SATV) and LOF (Open-Meteo)")

pa = st.selectbox("Price area (weather)", list(PRICE_AREAS.keys()),
                  index=list(PRICE_AREAS.keys()).index(st.session_state.get("price_area","NO1")))
df = fetch_openmeteo_hourly(pa, 2021)
if df.empty:
    st.error("No Open-Meteo-data.")
    st.stop()

tab1, tab2 = st.tabs(["SATV + SPC (temperature)", "LOF anomalies (precipitation)"])

with tab1:
    k = st.slider("SPC σ multiplier", 1.0, 5.0, 3.0, 0.5)
    cutoff = st.number_input("DCT cutoff (low-freq removed)", min_value=8, value=200, step=4)
    fig_satv, out_satv = satv_spc_plot(df, temp_col="temperature_2m", cutoff=int(cutoff), k_sigma=float(k))
    st.plotly_chart(fig_satv, use_container_width=True)
    st.dataframe(out_satv.head(20))

with tab2:
    cont = st.slider("LOF contamination (proportion)", 0.001, 0.05, 0.01, 0.001)
    nn   = st.number_input("LOF n_neighbors", min_value=5, value=20, step=1)
    fig_lof, out_lof = lof_anomaly_plot(df, var_col="precipitation", contamination=float(cont), n_neighbors=int(nn))
    st.plotly_chart(fig_lof, use_container_width=True)
    st.dataframe(out_lof.head(20))
