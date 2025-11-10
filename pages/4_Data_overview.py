import streamlit as st
import pandas as pd
import utils.io_utils as io   # <- riktig import (modul), ikke "from utils.io_utils import io"

st.set_page_config(page_title="Data overview", layout="wide")
st.title("Data overview")

# -----------------------------
# Elhub Production Data
# -----------------------------
st.header("Elhub Production Data")

# Fyll session_state hvis mulig (fra data/ eller prosjektrot)
io.ensure_elhub_in_session()

df_elhub = st.session_state.get("df_elhub_norm")
if df_elhub is None or df_elhub.empty:
    st.warning("Fant ikke Elhub-data automatisk. Last opp CSV under.")
    up = st.file_uploader("Upload Elhub CSV", type=["csv"], key="up_elhub")
    if up is not None:
        # Elhub er ofte semikolonseparert
        try:
            raw = pd.read_csv(up, sep=";", engine="python")
        except Exception:
            up.seek(0)
            raw = pd.read_csv(up)
        df_elhub = io.normalize_elhub_columns(raw)
        st.session_state["df_elhub_norm"] = df_elhub
    else:
        st.stop()

st.dataframe(df_elhub.head(200), use_container_width=True)
st.caption(f"Rows: {len(df_elhub):,}")

# -----------------------------
# Open-Meteo Hourly Data
# -----------------------------
st.header("Open-Meteo Hourly Data")

io.ensure_openmeteo_in_session()
df_om = st.session_state.get("hourly_dataframe")
if df_om is None or df_om.empty:
    st.warning("Fant ikke Open-Meteo-data automatisk. Last opp CSV under.")
    up2 = st.file_uploader("Upload Open-Meteo CSV", type=["csv"], key="up_openmeteo")
    if up2 is not None:
        # Prøv med parse_dates; fall tilbake uten om nødvendig
        try:
            raw2 = pd.read_csv(up2, parse_dates=["date", "time"], infer_datetime_format=True)
        except Exception:
            up2.seek(0)
            raw2 = pd.read_csv(up2)
        # Sørg for at vi har en 'date'-kolonne i datetime (UTC)
        if "date" not in raw2.columns and "time" in raw2.columns:
            raw2 = raw2.rename(columns={"time": "date"})
        raw2["date"] = pd.to_datetime(raw2["date"], errors="coerce", utc=True)
        df_om = raw2
        st.session_state["hourly_dataframe"] = df_om
    else:
        st.stop()

st.dataframe(df_om.head(200), use_container_width=True)
st.caption(f"Rows: {len(df_om):,}")

# -----------------------------
# Mini demo: first-month sparkline (Open-Meteo)
# -----------------------------
with st.expander("Mini demo: first-month sparkline (Open-Meteo)"):
    try:
        if "date" in df_om.columns:
            # Finn første måned som faktisk finnes i dataserien
            first_period = df_om["date"].dt.to_period("M").min()
            first_month_mask = df_om["date"].dt.to_period("M").eq(first_period)
            first_month = df_om.loc[first_month_mask].copy()

            if not first_month.empty:
                # Lag én rad per ikke-tidskolonne, med verdiliste for LineChartColumn
                for col in first_month.columns:
                    if col == "date":
                        continue
                    series_vals = pd.to_numeric(first_month[col], errors="coerce").tolist()
                    df_var = pd.DataFrame({"Variable": [col], "Series": [series_vals]})
                    st.data_editor(
                        df_var,
                        column_config={
                            "Variable": st.column_config.TextColumn(label="Variable Name", width="medium"),
                            "Series": st.column_config.LineChartColumn(
                                "Value",
                                help=f"Row-wise sparkline for {col}",
                                width="large",
                            ),
                        },
                        hide_index=True,
                    )
            else:
                st.info("Fant ingen rader i første måned av dataserien.")
        else:
            st.info("Fant ikke kolonnen 'date' i Open-Meteo-data.")
    except Exception as e:
        st.exception(e)
