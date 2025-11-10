import streamlit as st
import pandas as pd
from analysis_utils import satv_spc_plot, lof_anomaly_plot

st.set_page_config(page_title="Outliers & LOF", layout="wide")
st.title("Outlier/SPC (SATV) and LOF (Open-Meteo)")
st.caption("Seasonally adjusted temperature outliers and precipitation LOF anomalies.")

if "hourly_dataframe" not in st.session_state:
    st.error("Open-Meteo data not loaded. Go to the area/year page and trigger the API fetch.")
else:
    df = st.session_state["hourly_dataframe"].copy()

    tab1, tab2 = st.tabs(["SATV + SPC (temperature)", "LOF anomalies (precipitation)"])

    with tab1:
        k = st.slider("SPC σ multiplier", 1.0, 5.0, 3.0, 0.5)
        cutoff = st.number_input("DCT cutoff (low-freq removed)", min_value=8, value=200, step=4)
        fig_satv, out_satv = satv_spc_plot(df, temp_col="temperature_2m", cutoff=int(cutoff), k_sigma=float(k))
        st.plotly_chart(fig_satv, width="stretch")
        st.write("SATV outliers (head):")
        st.dataframe(out_satv.head(20))

    with tab2:
        cont = st.slider("LOF contamination (proportion)", 0.001, 0.05, 0.01, 0.001)
        nn   = st.number_input("LOF n_neighbors", min_value=5, value=20, step=1)
        fig_lof, out_lof = lof_anomaly_plot(df, var_col="precipitation", contamination=float(cont), n_neighbors=int(nn))
        st.plotly_chart(fig_lof, width="stretch")
        st.write("LOF anomalies (head):")
        st.dataframe(out_lof.head(20))
