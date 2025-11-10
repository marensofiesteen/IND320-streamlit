import streamlit as st
import utils.io_utils as io

st.set_page_config(page_title="Weather Data App", layout="wide")
st.title("Welcome to the Weather Data App")
st.write("Use the menu on the left to navigate between pages.")

# Last data inn i session_state (én gang)
io.ensure_elhub_in_session()
io.ensure_openmeteo_in_session()

# Diagnose (skru av når alt funker)
with st.expander("🔧 Diagnose (midlertidig)"):
    st.write("CWD:", st.experimental_get_query_params())  # bare som placeholder
    st.write("Elhub df:", "OK ✅" if "df_elhub_norm" in st.session_state and not st.session_state["df_elhub_norm"].empty else "Mangler ❌")
    if "__elhub_error__" in st.session_state:
        st.error(st.session_state["__elhub_error__"])
    st.write("Open-Meteo df:", "OK ✅" if "hourly_dataframe" in st.session_state and not st.session_state["hourly_dataframe"].empty else "Mangler ❌")
    if "__openmeteo_error__" in st.session_state:
        st.error(st.session_state["__openmeteo_error__"])

st.success("Hvis begge står som OK ✅ over, kan du åpne sidene i menyen til venstre.")