import streamlit as st
from db_mongo import fetch_year
from analysis_utils import stl_decompose_production, spectrogram_production
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS


st.set_page_config(page_title="STL & Spectrogram", layout="wide")
st.title("STL and Spectrogram (Elhub production)")

areas = ["NO1","NO2","NO3","NO4","NO5"]
area = st.selectbox("Price Area", areas, index=areas.index(st.session_state.get("price_area","NO1")))
groups = ["hydro","wind","thermal","solar","other"]
group = st.selectbox("Production Group", groups, index=groups.index((st.session_state.get("groups",groups)+["hydro"])[0]))

df = fetch_year(area, group)
if df.empty:
    st.error("No Elhub-data found in MongoDB for selected area/group.")
    st.stop()

tab1, tab2 = st.tabs(["STL", "Spectrogram"])

with tab1:
    period = st.number_input("Period length (hours)", min_value=24, value=168, step=24)
    trend_default = max(31*24, int(period) + 24)
    trend = st.number_input("Trend smoother (hours)", min_value=int(period)+1, value=trend_default, step=24)
    try:
        fig_stl, _ = stl_decompose_production(
            df.rename(columns={"datetime":"datetime","quantitykwh":"quantitykwh"}),
            area, group, period_length=int(period), trend_smoother=int(trend),
        )
        st.plotly_chart(fig_stl, use_container_width=True)
    except Exception as e:
        st.error(f"STL error: {e}")

with tab2:
    wl = st.number_input("Window length (hours)", min_value=32, value=168, step=8)
    ov = st.slider("Window overlap", 0.0, 0.9, 0.5, 0.05)
    try:
        fig_spec, _ = spectrogram_production(df, area, group, window_length_hours=int(wl), window_overlap=float(ov))
        st.plotly_chart(fig_spec, use_container_width=True)
    except Exception as e:
        st.error(f"Spectrogram feilet: {e}")
