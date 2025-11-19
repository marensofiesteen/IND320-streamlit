import streamlit as st
from db_mongo import fetch_year
from analysis_utils import stl_decompose_production, spectrogram_production
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS

st.set_page_config(page_title="STL & Spectrogram", layout="wide")
st.title("STL and Spectrogram (Elhub production)")

# --- Cached Elhub data loader (addresses the caching comment) ---
@st.cache_data(show_spinner=False)
def get_elhub_data(area: str, group: str):
    """Fetch one year of Elhub data for a given area and production group."""
    return fetch_year(area, group)


# --- Global selection of price area and production group ---
areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]

default_area = st.session_state.get("price_area", "NO1")
if default_area not in areas:
    default_area = "NO1"

area = st.selectbox(
    "Price area",
    areas,
    index=areas.index(default_area),
)

# Use global groups if available, otherwise fall back to full list
groups = ["hydro", "wind", "thermal", "solar", "other"]
session_groups = st.session_state.get("groups", groups)
if not session_groups:
    session_groups = groups

default_group = st.session_state.get("production_group", session_groups[0])
if default_group not in groups:
    default_group = session_groups[0]

group = st.selectbox(
    "Production group",
    groups,
    index=groups.index(default_group),
)

# Keep global selections in sync so other pages can use them
st.session_state["price_area"] = area
st.session_state["production_group"] = group

# --- Load data (cached) ---
df = get_elhub_data(area, group)

if df.empty:
    st.error("No Elhub data found in MongoDB for the selected area/group.")
    st.stop()

tab1, tab2 = st.tabs(["STL", "Spectrogram"])

# ----------------- TAB 1: STL -----------------
with tab1:
    st.subheader("STL decomposition")

    # Slider for seasonal period
    period = st.slider(
        "Seasonal period (hours)",
        min_value=24,
        max_value=24 * 28,   # up to 4 weeks
        value=168,
        step=24,
        help="Typical choice for hourly data is 168 hours (1 week).",
    )

    # Default trend window: at least period + some buffer
    trend_default = max(31 * 24, int(period) + 24)

    trend = st.slider(
        "Trend smoother (hours)",
        min_value=int(period) + 1,
        max_value=24 * 180,   # up to ~6 months of smoothing
        value=trend_default,
        step=24,
        help="Larger values give a smoother trend curve.",
    )

    # Spinner while STL is computed
    with st.spinner("Computing STL decomposition..."):
        try:
            fig_stl, _ = stl_decompose_production(
                df.rename(columns={"datetime": "datetime", "quantitykwh": "quantitykwh"}),
                area,
                group,
                period_length=int(period),
                trend_smoother=int(trend),
            )
            st.plotly_chart(fig_stl, use_container_width=True)
        except Exception as e:
            st.error(f"STL error: {e}")

# ----------------- TAB 2: Spectrogram -----------------
with tab2:
    st.subheader("Spectrogram")

    wl = st.slider(
        "Window length (hours)",
        min_value=32,
        max_value=24 * 60,   # up to ~2 months
        value=168,
        step=8,
        help="Length of the STFT analysis window.",
    )

    ov = st.slider(
        "Window overlap (fraction)",
        min_value=0.0,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Fractional overlap between consecutive windows.",
    )

    with st.spinner("Computing spectrogram..."):
        try:
            fig_spec, _ = spectrogram_production(
                df,
                area,
                group,
                window_length_hours=int(wl),
                window_overlap=float(ov),
            )
            st.plotly_chart(fig_spec, use_container_width=True)
        except Exception as e:
            st.error(f"Spectrogram error: {e}")


