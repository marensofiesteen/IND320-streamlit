import streamlit as st
import utils.io_utils as io
from analysis_utils import stl_decompose_production, spectrogram_production

st.set_page_config(page_title="STL & Spectrogram", layout="wide")
st.title("STL and Spectrogram (Elhub production)")
st.caption("Select parameters and visualize seasonality and time-frequency content.")

# --- Sørg for data i session ---
io.ensure_elhub_in_session()
df = st.session_state.get("df_elhub_norm")
if df is None or df.empty:
    st.error("Elhub-data mangler. Legg CSV i 'data/' og prøv igjen.")
    st.stop()

# --- UI: valg av område/gruppe og parametere ---
colA, colB, colC, colD = st.columns(4)
with colA:
    areas = sorted(df["pricearea"].dropna().unique().tolist())
    area = st.selectbox("Price Area", areas)

with colB:
    groups = sorted(df.loc[df["pricearea"] == area, "productiongroup"].dropna().unique().tolist())
    if not groups:
        st.error(f"Ingen produksjonsgrupper for område {area}.")
        st.stop()
    group = st.selectbox("Production Group", groups)

with colC:
    period = st.number_input("Period length (hours)", min_value=24, value=168, step=24)

with colD:
    trend_default = max(31*24, int(period) + 24)
    trend = st.number_input("Trend smoother (hours)", min_value=int(period)+1, value=trend_default, step=24)

tab1, tab2 = st.tabs(["STL", "Spectrogram"])

# --- Tab 1: STL ---
with tab1:
    try:
        fig_stl, _ = stl_decompose_production(
            df, area, group,
            period_length=int(period),
            trend_smoother=int(trend),
        )
        st.plotly_chart(fig_stl, use_container_width=True)
    except Exception as e:
        st.error(f"STL feilet: {e}")

# --- Tab 2: Spectrogram ---
with tab2:
    wl = st.number_input("Window length (hours)", min_value=32, value=168, step=8)
    ov = st.slider("Window overlap", 0.0, 0.9, 0.5, 0.05)
    try:
        fig_spec, _ = spectrogram_production(
            df, area, group,
            window_length_hours=int(wl),
            window_overlap=float(ov),
        )
        st.plotly_chart(fig_spec, use_container_width=True)
    except Exception as e:
        st.error(f"Spectrogram feilet: {e}")

