
import streamlit as st
import pandas as pd
import plotly.express as px
import utils.io_utils as io   # <- viktig: ikke fra analysis_utils

st.set_page_config(page_title="Plots", layout="wide")
st.header("Time series exploration of weather data")

# 1) Sørg for data i session_state
io.ensure_openmeteo_in_session()

df = st.session_state.get("hourly_dataframe")
if df is None or df.empty:
    st.error("Open-Meteo-data mangler. Sjekk at 'data/open-meteo-subset.csv' finnes, eller last opp via Data overview-siden.")
    st.stop()

# 2) Standardiser kolonner: vi bruker 'date' som tidskolonne i hele appen
if "date" not in df.columns:
    # ekstra fallbacks – skulle ikke trenges etter io_utils, men trygt å ha
    for alt in ["time", "Time", "timestamp", "datetime"]:
        if alt in df.columns:
            df = df.rename(columns={alt: "date"})
            break
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

# 3) Lag år-måned for slider
df["year_month"] = df["date"].dt.strftime("%Y-%m")

# 4) Velg variabel
# Ekskluder tidskolonner og hjelpekolonner
exclude = {"date", "year_month"}
meteo_options = [c for c in df.columns if c not in exclude]
# Hvis noen kolonner er strings med tall, la oss konvertere dem
for c in meteo_options:
    if df[c].dtype == "object":
        df[c] = pd.to_numeric(df[c], errors="ignore")

option_meteo = st.selectbox(
    "Step 1: Select one variable",
    options=meteo_options + ["Show all"],
    index=0
)

# 5) Velg måned
month_list = sorted(df["year_month"].dropna().unique().tolist())
if not month_list:
    st.warning("Fant ingen gyldige datoer i datasettet.")
    st.stop()

option_month = st.select_slider(
    "Step 2: Select a month",
    options=month_list,
    value=month_list[0],
)
st.caption(f"Selected: {option_meteo} and {option_month}")

# 6) Filtrer og plott
df_filtered = df[df["year_month"] <= option_month].copy()
start_month = df_filtered["year_month"].min()

if option_meteo != "Show all":
    if option_meteo not in df_filtered.columns:
        st.error(f"Kolonnen '{option_meteo}' finnes ikke i datasettet.")
        st.stop()
    fig = px.line(
        df_filtered,
        x="date",
        y=option_meteo,
        title=f"{option_meteo} from {start_month} to {option_month}",
        labels={"date": "Date", option_meteo: option_meteo},
    )
else:
    df_melt = df_filtered.drop(columns=["year_month"]).melt(
        id_vars="date", var_name="variable", value_name="value"
    )
    fig = px.line(
        df_melt,
        x="date",
        y="value",
        color="variable",
        title=f"All variables from {start_month} to {option_month}",
        labels={"date": "Date", "value": "Value", "variable": "Variable"},
    )

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
