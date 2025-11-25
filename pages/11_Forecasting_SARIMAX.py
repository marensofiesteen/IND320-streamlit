import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pymongo import MongoClient, errors as mongo_errors
from statsmodels.tsa.statespace.sarimax import SARIMAX
import requests

from utils.navigation import sidebar_navigation

# Local meny for this group
sidebar_navigation("Forecasting")

st.set_page_config(page_title="Forecasting – SARIMAX", layout="wide")
st.title("Forecasting of energy production and consumption (SARIMAX)")

# -----------------------------
#  GLOBAL LIMITS (for PC-helse)
# -----------------------------
MAX_TRAIN_DAYS = 365        # maks lengde på treningsperiode (1 år)
MAX_HORIZON_DAYS = 30       # maks forecast-horisont i dager
HIST_HORIZON_DAYS = 7       # hvor mye historikk som vises i plottet (siste X dager)


# ---------------------------------------------------------
#  Data loaders
# ---------------------------------------------------------


@st.cache_data(show_spinner=True)
def load_elhub_collection(collection_name: str) -> pd.DataFrame:
    """
    Load a MongoDB collection (production_data or consumption_data) into a DataFrame.
    Includes basic error handling.
    """
    try:
        mongo_secrets = st.secrets["mongo"]
        uri = mongo_secrets["uri"]
        db_name = mongo_secrets["db"]
    except Exception as e:
        st.error(f"Missing or invalid MongoDB secrets: {e}")
        return pd.DataFrame()

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[db_name]
        coll = db[collection_name]
        docs = list(coll.find({}, {"_id": 0}))
    except mongo_errors.ServerSelectionTimeoutError:
        st.error("Could not reach MongoDB server. Check your network or MongoDB settings.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error while loading data from MongoDB: {e}")
        return pd.DataFrame()

    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df.columns = [c.lower() for c in df.columns]

    if "starttime" in df.columns:
        df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
        df = df.dropna(subset=["starttime"])

    return df


PRICE_AREA_LOCATIONS = {
    "NO1": {"city": "Oslo",         "lat": 59.9139,  "lon": 10.7522},
    "NO2": {"city": "Kristiansand", "lat": 58.1467,  "lon": 7.9956},
    "NO3": {"city": "Trondheim",    "lat": 63.4305,  "lon": 10.3951},
    "NO4": {"city": "Tromsø",       "lat": 69.6492,  "lon": 18.9553},
    "NO5": {"city": "Bergen",       "lat": 60.39299, "lon": 5.32415},
}


@st.cache_data(show_spinner=True)
def load_weather_from_open_meteo(price_area: str,
                                 start_time: pd.Timestamp,
                                 end_time: pd.Timestamp) -> pd.DataFrame:
    """
    Hent værdata fra Open-Meteo ERA5 for en prisområde-representativ by
    og en gitt tidsperiode. Brukes som exogene variabler i SARIMAX.
    """
    if price_area not in PRICE_AREA_LOCATIONS:
        return pd.DataFrame()

    loc = PRICE_AREA_LOCATIONS[price_area]
    lat = loc["lat"]
    lon = loc["lon"]

    base_url = "https://archive-api.open-meteo.com/v1/era5"

    start_date = start_time.floor("D")
    end_date = end_time.ceil("D")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
        ],
        "timezone": "Europe/Oslo",
    }

    try:
        resp = requests.get(base_url, params=params, timeout=60)
    except requests.exceptions.RequestException:
        return pd.DataFrame()

    if resp.status_code != 200:
        return pd.DataFrame()

    try:
        data = resp.json()
    except ValueError:
        return pd.DataFrame()

    if "hourly" not in data or "time" not in data["hourly"]:
        return pd.DataFrame()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.rename(columns={"time": "date"})

    clean_cols = {}
    for c in df.columns:
        if c == "date":
            clean_cols[c] = "date"
        else:
            clean_cols[c] = c.strip().lower()
    df = df.rename(columns=clean_cols)

    df = df.sort_values("date")
    return df


# ---------------------------------------------------------
#  Load energy data
# ---------------------------------------------------------


with st.spinner("Loading energy data from MongoDB ..."):
    df_prod = load_elhub_collection("production_data")
    df_cons = load_elhub_collection("consumption_data")

if df_prod.empty and df_cons.empty:
    st.error("No energy data available from MongoDB (both production and consumption are empty).")
    st.stop()

if df_prod.empty:
    st.warning("Production data not available.")
if df_cons.empty:
    st.warning("Consumption data not available.")


# ---------------------------------------------------------
#  Build shared year / area info
# ---------------------------------------------------------

year_series_list = []
if not df_prod.empty and "starttime" in df_prod.columns:
    year_series_list.append(df_prod["starttime"].dt.year)
if not df_cons.empty and "starttime" in df_cons.columns:
    year_series_list.append(df_cons["starttime"].dt.year)

if not year_series_list:
    st.error("Could not determine available years from energy data.")
    st.stop()

years = sorted(pd.concat(year_series_list).unique())
year_min, year_max = int(min(years)), int(max(years))

area_set = set()
if not df_prod.empty and "pricearea" in df_prod.columns:
    area_set.update(df_prod["pricearea"].dropna().unique())
if not df_cons.empty and "pricearea" in df_cons.columns:
    area_set.update(df_cons["pricearea"].dropna().unique())

available_areas = sorted(area_set)

# ---------------------------------------------------------
#  Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("Data selection")

energy_mode = st.sidebar.radio(
    "Energy mode",
    [
        "Production",
        "Consumption",
        "Both (production + consumption + net load)",
    ],
    index=0,
)

price_area = st.sidebar.selectbox("Price area", available_areas)

group = None
group_col = None
title_prefix = ""

# For Production / Consumption: velg gruppe
if energy_mode == "Production":
    if df_prod.empty:
        st.sidebar.error("Production data not available.")
        st.stop()
    df_energy_all = df_prod.copy()
    group_col = "productiongroup"
    title_prefix = "Production"

    if group_col not in df_energy_all.columns:
        st.error(f"Column '{group_col}' missing from production data.")
        st.stop()

    available_groups = sorted(df_energy_all[group_col].dropna().unique())
    group = st.sidebar.selectbox("Energy group", available_groups)

elif energy_mode == "Consumption":
    if df_cons.empty:
        st.sidebar.error("Consumption data not available.")
        st.stop()
    df_energy_all = df_cons.copy()
    group_col = "consumptiongroup"
    title_prefix = "Consumption"

    if group_col not in df_energy_all.columns:
        st.error(f"Column '{group_col}' missing from consumption data.")
        st.stop()

    available_groups = sorted(df_energy_all[group_col].dropna().unique())
    group = st.sidebar.selectbox("Energy group", available_groups)

else:
    # Both-mode: vi aggregerer over alle grupper, derfor ingen group-select
    title_prefix = "Production & Consumption (aggregated)"


st.sidebar.subheader("Training period")
train_start_year, train_end_year = st.sidebar.select_slider(
    "Training years (inclusive)",
    options=years,
    value=(year_min, year_max - 1 if year_max > year_min else year_max),
)

# Forecast horizon i dager (begrenset)
horizon_days = st.sidebar.slider(
    "Forecast horizon (days)",
    1,
    MAX_HORIZON_DAYS,
    7,
)
horizon_hours = horizon_days * 24

st.sidebar.subheader("SARIMAX parameters")
p = st.sidebar.number_input("AR order (p)", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("Diff order (d)", min_value=0, max_value=2, value=0, step=1)
q = st.sidebar.number_input("MA order (q)", min_value=0, max_value=5, value=1, step=1)

P = st.sidebar.number_input("Seasonal AR (P)", min_value=0, max_value=5, value=0, step=1)
D = st.sidebar.number_input("Seasonal diff (D)", min_value=0, max_value=2, value=0, step=1)
Q = st.sidebar.number_input("Seasonal MA (Q)", min_value=0, max_value=5, value=0, step=1)
s = st.sidebar.number_input("Seasonal period (s, hours)", min_value=1, max_value=24 * 31, value=24, step=1)

st.sidebar.subheader("Exogenous variables (weather)")
use_exog = st.sidebar.checkbox("Use weather as exogenous variables", value=False)
exog_choices = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
]
selected_exog = []
if use_exog:
    selected_exog = st.sidebar.multiselect(
        "Select weather variables",
        options=exog_choices,
        default=["temperature_2m"],
    )

# I BOTH-modus: vi skrur av exog (forenkling)
if energy_mode.startswith("Both") and use_exog:
    st.sidebar.warning("Weather exogenous variables are only supported for single-series mode. Disabling for 'Both'.")
    use_exog = False
    selected_exog = []


# ---------------------------------------------------------
#  Common time bounds
# ---------------------------------------------------------

# Treningsperiode basert på year-slider
train_start = pd.Timestamp(f"{train_start_year}-01-01")
train_end = pd.Timestamp(f"{train_end_year}-12-31 23:00:00")

# ---------------------------------------------------------
#  Helper for single-series mode
# ---------------------------------------------------------

def run_single_series_mode():
    """Kjør SARIMAX på én serie (production ELLER consumption)."""
    global title_prefix  # for bruk i teksten nederst

    df_energy_all_local = df_energy_all.copy()
    df_energy_all_local["year"] = df_energy_all_local["starttime"].dt.year

    mask = (
        (df_energy_all_local["pricearea"] == price_area)
        & (df_energy_all_local[group_col] == group)
        & (df_energy_all_local["year"] >= train_start_year)
        & (df_energy_all_local["year"] <= year_max)
    )
    df_energy_sel = df_energy_all_local.loc[mask].copy()

    if df_energy_sel.empty:
        st.warning("No energy data for the selected filters.")
        st.stop()

    ts = (
        df_energy_sel
        .set_index("starttime")
        .sort_index()
        .resample("1H")["quantitykwh"]
        .sum()
    )

    if ts.empty:
        st.warning("Time series is empty after resampling.")
        st.stop()

    # Juster treningsstart/-slutt til faktisk data
    train_start_eff = max(train_start, ts.index.min())
    train_end_eff = min(train_end, ts.index.max())

    if train_start_eff >= train_end_eff:
        st.warning("Training period is invalid after intersecting with available data.")
        st.stop()

    actual_train_days = (train_end_eff - train_start_eff).days + 1
    if actual_train_days > MAX_TRAIN_DAYS:
        st.error(
            f"Training period is too long ({actual_train_days} days). "
            f"Please select at most {MAX_TRAIN_DAYS} days to avoid memory issues."
        )
        st.stop()

    y_train = ts.loc[train_start_eff:train_end_eff]

    forecast_start = train_end_eff + pd.Timedelta(hours=1)
    forecast_end = forecast_start + pd.Timedelta(hours=horizon_hours - 1)
    y_true = ts.loc[forecast_start:forecast_end]

    # Exogene variabler
    exog_train = None
    exog_forecast = None

    if use_exog and selected_exog:
        with st.spinner("Downloading weather data for exogenous variables ..."):
            weather_df = load_weather_from_open_meteo(
                price_area=price_area,
                start_time=train_start_eff,
                end_time=forecast_end,
            )

        if weather_df.empty:
            st.warning("Could not load weather data for exogenous variables. Continuing without exog.")
        else:
            weather_ts = (
                weather_df
                .set_index("date")
                .sort_index()
                .resample("1H")
                .mean()
            )
            weather_ts = weather_ts[selected_exog]
            weather_ts = weather_ts.reindex(ts.index).interpolate().ffill().bfill()

            exog_train = weather_ts.loc[train_start_eff:train_end_eff]
            exog_forecast = weather_ts.loc[forecast_start:forecast_end]

    # ------- Fit + forecast -------
    st.subheader("Model fitting and forecasting")

    if st.button("Run SARIMAX forecast"):
        with st.spinner("Running SARIMAX model – this may take a while ..."):
            progress = st.progress(0, text="Initialising SARIMAX model ...")
            try:
                model = SARIMAX(
                    y_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    exog=exog_train if use_exog and exog_train is not None else None,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                progress.progress(40, text="Fitting SARIMAX model ...")

                results = model.fit(disp=False)

                progress.progress(70, text="Generating forecast ...")
                forecast_res = results.get_forecast(
                    steps=horizon_hours,
                    exog=exog_forecast if use_exog and exog_forecast is not None else None,
                )

                forecast_mean = forecast_res.predicted_mean
                forecast_ci = forecast_res.conf_int()

                df_forecast = pd.DataFrame({
                    "time": forecast_mean.index,
                    "forecast": forecast_mean.values,
                    "lower": forecast_ci.iloc[:, 0].values,
                    "upper": forecast_ci.iloc[:, 1].values,
                }).set_index("time")

                progress.progress(100, text="Forecast completed.")
            except Exception as e:
                progress.empty()
                st.error(f"Error while fitting SARIMAX or generating forecast: {e}")
                st.stop()
            finally:
                progress.empty()

            # ------- Plot -------
            hist_horizon_hours = HIST_HORIZON_DAYS * 24
            y_hist = y_train[-hist_horizon_hours:] if len(y_train) > 0 else y_train

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=y_hist.index,
                    y=y_hist.values,
                    mode="lines",
                    name="History (train)",
                )
            )

            if not y_true.empty:
                fig.add_trace(
                    go.Scatter(
                        x=y_true.index,
                        y=y_true.values,
                        mode="lines",
                        name="Actual (future)",
                        line=dict(dash="dot"),
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=df_forecast.index,
                    y=df_forecast["forecast"],
                    mode="lines",
                    name="Forecast",
                )
            )

            ci_x = df_forecast.index.tolist() + df_forecast.index[::-1].tolist()
            ci_y = (
                df_forecast["upper"].tolist()
                + df_forecast["lower"][::-1].tolist()
            )

            fig.add_trace(
                go.Scatter(
                    x=ci_x,
                    y=ci_y,
                    fill="toself",
                    fillcolor="rgba(0, 0, 255, 0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="Confidence interval",
                )
            )

            # y-akse-zoom
            series_for_scale = []
            if not y_hist.empty:
                series_for_scale.append(y_hist)
            if not y_true.empty:
                series_for_scale.append(y_true)
            if not df_forecast["forecast"].empty:
                series_for_scale.append(df_forecast["forecast"])

            if series_for_scale:
                y_for_scale = pd.concat(series_for_scale)
                y_min = y_for_scale.min()
                y_max = y_for_scale.max()
                padding = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
                fig.update_yaxes(range=[y_min - padding, y_max + padding])

            fig.update_layout(
                title=dict(
                    text=f"SARIMAX forecast – {title_prefix.lower()} quantitykwh, {price_area}, group {group}",
                    y=0.98,
                    x=0.5,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=20)
                ),
                margin=dict(t=80),
                xaxis_title="Time",
                yaxis_title="Energy (kWh)",
                hovermode="x unified",
            )


            st.plotly_chart(fig, use_container_width=True)

            # Summary-tekst
            st.markdown(
                f"""
                **Model summary (short):**

                - Data type: **{title_prefix}**
                - Price area: **{price_area}**
                - Group: **{group}**
                - Training period: `{train_start_eff.date()} – {train_end_eff.date()}`
                - Training length (days): **{actual_train_days}** (max {MAX_TRAIN_DAYS})
                - Forecast horizon: **{horizon_days} days** (≈ {horizon_hours} hours, max {MAX_HORIZON_DAYS} days)
                - SARIMAX order: ({p}, {d}, {q})
                - Seasonal order: ({P}, {D}, {Q}, {s})
                - Exogenous variables: {", ".join(selected_exog) if use_exog and selected_exog else "None"}
                """
            )

    else:
        st.subheader("Model fitting and forecasting")
        st.info(
            "Adjust the settings in the sidebar and click **Run SARIMAX forecast** "
            "to generate a forecast. The plot will show only the last "
            f"{HIST_HORIZON_DAYS} days of training data for readability."
        )


# ---------------------------------------------------------
#  Helper for BOTH-mode
# ---------------------------------------------------------

def run_both_mode():
    """Kjør SARIMAX på aggregert produksjon + forbruk + nettolast."""
    # --- Bygg timeserier for prod + cons (aggregert over alle grupper) ---
    if df_prod.empty or df_cons.empty:
        st.error("Both production and consumption data are needed for 'Both' mode.")
        st.stop()

    dfp = df_prod[df_prod["pricearea"] == price_area].copy()
    dfc = df_cons[df_cons["pricearea"] == price_area].copy()

    if dfp.empty or dfc.empty:
        st.error("Missing either production or consumption data for this price area.")
        st.stop()

    ts_prod = (
        dfp.set_index("starttime")
        .sort_index()
        .resample("1H")["quantitykwh"]
        .sum()
    )
    ts_cons = (
        dfc.set_index("starttime")
        .sort_index()
        .resample("1H")["quantitykwh"]
        .sum()
    )

    if ts_prod.empty or ts_cons.empty:
        st.error("Time series for production or consumption is empty after resampling.")
        st.stop()

    # Felles tidsakse
    common_index = ts_prod.index.intersection(ts_cons.index)
    if common_index.empty:
        st.error("No overlapping timestamps between production and consumption.")
        st.stop()

    ts_prod = ts_prod.reindex(common_index)
    ts_cons = ts_cons.reindex(common_index)

    # Trening innenfor valgt år, men begrenset til det som faktisk finnes
    train_start_eff = max(train_start, common_index.min())
    train_end_eff = min(train_end, common_index.max())

    if train_start_eff >= train_end_eff:
        st.warning("Training period is invalid after intersecting with available data.")
        st.stop()

    actual_train_days = (train_end_eff - train_start_eff).days + 1
    if actual_train_days > MAX_TRAIN_DAYS:
        st.error(
            f"Training period is too long ({actual_train_days} days). "
            f"Please select at most {MAX_TRAIN_DAYS} days to avoid memory issues."
        )
        st.stop()

    mask_train = (common_index >= train_start_eff) & (common_index <= train_end_eff)
    idx_train = common_index[mask_train]

    y_prod_train = ts_prod.reindex(idx_train)
    y_cons_train = ts_cons.reindex(idx_train)

    if y_prod_train.empty or y_cons_train.empty:
        st.error("No overlapping production/consumption data in the selected training period.")
        st.stop()

    net_hist_full = y_cons_train - y_prod_train

    forecast_start = train_end_eff + pd.Timedelta(hours=1)
    forecast_end = forecast_start + pd.Timedelta(hours=horizon_hours - 1)

    # ------- Fit modeller -------
    st.subheader("Model fitting and forecasting – Production, Consumption & Net load")

    if st.button("Run SARIMAX forecast (both)"):
        with st.spinner("Running SARIMAX models for production and consumption ..."):
            progress = st.progress(0, text="Initialising SARIMAX models ...")
            try:
                model_prod = SARIMAX(
                    y_prod_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                model_cons = SARIMAX(
                    y_cons_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                progress.progress(30, text="Fitting production model ...")
                res_prod = model_prod.fit(disp=False)

                progress.progress(60, text="Fitting consumption model ...")
                res_cons = model_cons.fit(disp=False)

                progress.progress(80, text="Generating forecasts ...")
                fc_prod_res = res_prod.get_forecast(steps=horizon_hours)
                fc_cons_res = res_cons.get_forecast(steps=horizon_hours)

                fc_prod_mean = fc_prod_res.predicted_mean
                fc_cons_mean = fc_cons_res.predicted_mean

                fc_prod_ci = fc_prod_res.conf_int()
                fc_cons_ci = fc_cons_res.conf_int()

                net_fc = fc_cons_mean - fc_prod_mean

                progress.progress(100, text="Forecast completed.")
            except Exception as e:
                progress.empty()
                st.error(f"Error while fitting SARIMAX or generating forecasts: {e}")
                st.stop()
            finally:
                progress.empty()

            # ------- Plot -------
            hist_horizon_hours = HIST_HORIZON_DAYS * 24
            y_prod_hist = y_prod_train[-hist_horizon_hours:]
            y_cons_hist = y_cons_train[-hist_horizon_hours:]
            net_hist = net_hist_full[-hist_horizon_hours:]

            fig = go.Figure()

            # Historikk
            fig.add_trace(go.Scatter(
                x=y_prod_hist.index,
                y=y_prod_hist.values,
                mode="lines",
                name="Historical production",
                line=dict(color="green")
            ))
            fig.add_trace(go.Scatter(
                x=y_cons_hist.index,
                y=y_cons_hist.values,
                mode="lines",
                name="Historical consumption",
                line=dict(color="red")
            ))
            fig.add_trace(go.Scatter(
                x=net_hist.index,
                y=net_hist.values,
                mode="lines+markers",
                name="Historical net load (consumption − production)",
                line=dict(color="gray", dash="dot"),
                opacity=0.7
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=fc_prod_mean.index,
                y=fc_prod_mean.values,
                mode="lines",
                name="Forecast production",
                line=dict(color="green", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=fc_cons_mean.index,
                y=fc_cons_mean.values,
                mode="lines",
                name="Forecast consumption",
                line=dict(color="red", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=net_fc.index,
                y=net_fc.values,
                mode="lines+markers",
                name="Forecast net load (consumption − production)",
                line=dict(color="gray", dash="dashdot"),
                opacity=0.9
            ))

            # CI production
            fig.add_trace(go.Scatter(
                x=fc_prod_mean.index.tolist() + fc_prod_mean.index[::-1].tolist(),
                y=fc_prod_ci.iloc[:, 0].tolist() + fc_prod_ci.iloc[:, 1][::-1].tolist(),
                fill="toself",
                fillcolor="rgba(0, 128, 0, 0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Confidence interval – production",
                showlegend=True,
            ))

            # CI consumption
            fig.add_trace(go.Scatter(
                x=fc_cons_mean.index.tolist() + fc_cons_mean.index[::-1].tolist(),
                y=fc_cons_ci.iloc[:, 0].tolist() + fc_cons_ci.iloc[:, 1][::-1].tolist(),
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Confidence interval – consumption",
                showlegend=True,
            ))

            # y-akse-zoom
            series_for_scale = [
                y_prod_hist,
                y_cons_hist,
                net_hist,
                fc_prod_mean,
                fc_cons_mean,
                net_fc,
            ]
            y_for_scale = pd.concat(series_for_scale)
            y_min = y_for_scale.min()
            y_max = y_for_scale.max()
            padding = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
            fig.update_yaxes(range=[y_min - padding, y_max + padding])

            fig.update_layout(
                height=550,
                hovermode="x unified",
                title=dict(
                    text=f"SARIMAX forecast – production, consumption & net load – {price_area}",
                    y=0.98,
                    x=0.5,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=20)
                ),
                margin=dict(t=80),
                xaxis_title="Time",
                yaxis_title="Energy (kWh)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"""
                **Model summary (short):**

                - Energy mode: **Production + Consumption + Net load (aggregated)**  
                - Price area: **{price_area}**
                - Training period: `{train_start_eff.date()} – {train_end_eff.date()}`
                - Training length (days): **{actual_train_days}** (max {MAX_TRAIN_DAYS})
                - Forecast horizon: **{horizon_days} days** (≈ {horizon_hours} hours, max {MAX_HORIZON_DAYS} days)
                - SARIMAX order (both models): ({p}, {d}, {q})
                - Seasonal order (both models): ({P}, {D}, {Q}, {s})
                - Exogenous variables: **None in 'Both' mode** (simplified implementation)
                """
            )

    else:
        st.info(
            "Adjust the settings in the sidebar and click **Run SARIMAX forecast (both)** "
            "to train separate models for production and consumption and plot net load as well."
        )


# ---------------------------------------------------------
#  MAIN DISPATCH
# ---------------------------------------------------------

if energy_mode in ("Production", "Consumption"):
    run_single_series_mode()
else:
    run_both_mode()

# ---------------------------------------------------------
#  Help text
# ---------------------------------------------------------

with st.expander("How to interpret this page"):
    st.markdown(
        f"""
        - This page implements a **SARIMAX** model for forecasting energy
          production/consumption (`quantitykwh`) for a chosen
          price area (**{", ".join(available_areas)}**).
        - You select:
          - Training years (timeframe for fitting the model) – internally limited
            to max **{MAX_TRAIN_DAYS} days** to avoid performance issues.
          - Forecast horizon (in days), limited to max **{MAX_HORIZON_DAYS} days**.
          - SARIMAX parameters (p, d, q, P, D, Q, s).
          - Whether to include **weather variables** as exogenous variables
            (only for the single-series modes).
        - In **Production/Consumption mode**, the plot shows:
          - The last **{HIST_HORIZON_DAYS} days** of historical training data.
          - Actual future values (if available) as a dotted line.
          - The SARIMAX forecast with a shaded confidence interval.
        - In **Both mode**, the plot shows:
          - Historical and forecast **production**, **consumption** and
            **net load (consumption − production)**, plus confidence intervals
            for production and consumption.
        - Waiting time & error handling:
          - A progress bar and spinner indicate the steps of model initialisation, fitting and forecasting.
          - MongoDB/API issues, invalid time ranges and too-long training periods
            are caught and reported as friendly messages instead of raw Python errors.
        """
    )
