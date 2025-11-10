from pathlib import Path
import pandas as pd
import streamlit as st

# ---------- Elhub ----------
@st.cache_data
def load_elhub_csv() -> pd.DataFrame:
    """Finn Elhub-CSV både i /data og prosjektrot. Sniffer separator (; vs ,)."""
    here = Path(__file__).resolve()
    roots = [here] + list(here.parents)
    candidates = []
    for r in roots:
        candidates += [
            r / "data" / "Elhub_production_2021_all_areas.csv",
            r / "data" / "elhub_production_2021_all_areas.csv",
            r / "Elhub_production_2021_all_areas.csv",
            r / "elhub_production_2021_all_areas.csv",
        ]

    for p in candidates:
        if p.exists():
            # Prøv å lese med ;, men fall tilbake til , hvis det ble "én kolonne" eller mangler forventede felt
            def _read(path, sep):
                try:
                    return pd.read_csv(path, sep=sep, engine="python")
                except Exception:
                    return None

            df = _read(p, ";")
            bad = df is None or df.shape[1] == 1 or not set(
                c.lower() for c in df.columns
            ) & {"pricearea", "productiongroup", "quantitykwh", "starttime", "starttime".lower(), "starttime".upper()}
            if bad:
                df = pd.read_csv(p)  # default = komma
            return df

    raise FileNotFoundError("Fant ikke Elhub CSV i 'data/' eller prosjektrot.")


def normalize_elhub_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Tilpasset dine headere:
    endTime,lastUpdatedTime,priceArea,productionGroup,quantityKwh,startTime
    Bruker startTime som 'datetime' (UTC), og quantityKwh -> kWh.
    """
    df = df_in.copy()

    # Normaliser case
    rename_exact = {
        "priceArea": "pricearea",
        "productionGroup": "productiongroup",
        "quantityKwh": "quantitykwh",
        "startTime": "datetime",
        "endTime": "endtime",
        "lastUpdatedTime": "lastupdatedtime",
    }
    for k, v in rename_exact.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # Hvis ikke camelCase, prøv case-insensitivt
    cols_l = {c.lower(): c for c in df.columns}
    def has(col): return col in df.columns or col in cols_l
    def get(col): return col if col in df.columns else cols_l[col]

    required_any = [
        ("datetime", ["datetime", "starttime", "start_time"]),
        ("pricearea", ["pricearea", "price_area"]),
        ("productiongroup", ["productiongroup", "production_group"]),
        ("quantitykwh", ["quantitykwh", "quantity", "kwh"]),
    ]
    for std, alts in required_any:
        if std not in df.columns:
            for a in alts:
                if a in cols_l:
                    df = df.rename(columns={cols_l[a]: std})
                    break

    # Parse tid
    if has("datetime"):
        df["datetime"] = pd.to_datetime(df[get("datetime")], errors="coerce", utc=True)
    elif has("starttime"):
        df["datetime"] = pd.to_datetime(df[get("starttime")], errors="coerce", utc=True)
    else:
        raise KeyError("Fant ingen startTime/datetime-kolonne i Elhub-filen.")

    # Mengde til kWh
    if has("quantitykwh"):
        df["quantitykwh"] = pd.to_numeric(df[get("quantitykwh")], errors="coerce")
    else:
        raise KeyError("Fant ingen quantityKwh/quantity-kolonne i Elhub-filen.")

    # Områder og gruppe
    if has("pricearea"):
        df["pricearea"] = df[get("pricearea")]
    else:
        raise KeyError("Fant ingen priceArea/price_area-kolonne i Elhub-filen.")

    if has("productiongroup"):
        df["productiongroup"] = df[get("productiongroup")]
    else:
        raise KeyError("Fant ingen productionGroup/production_group-kolonne i Elhub-filen.")

    out = df[["pricearea", "productiongroup", "datetime", "quantitykwh"]].dropna(subset=["datetime"])
    return out.sort_values("datetime")


def ensure_elhub_in_session():
    if "df_elhub_norm" in st.session_state and not st.session_state["df_elhub_norm"].empty:
        return
    try:
        raw = load_elhub_csv()
        st.session_state["df_elhub_norm"] = normalize_elhub_columns(raw)
    except Exception as e:
        st.session_state["df_elhub_norm"] = pd.DataFrame()
        st.session_state["__elhub_error__"] = str(e)

# ---------- Open-Meteo ----------
@st.cache_data
def load_openmeteo_csv() -> pd.DataFrame:
    here = Path(__file__).resolve()
    roots = [here] + list(here.parents)
    candidates = []
    for r in roots:
        candidates += [r/"data"/"open-meteo-subset.csv", r/"open-meteo-subset.csv"]

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)  # komma
            # Map "Time" -> date
            if "Time" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"Time": "date"})
            # Lag rene navn for resten (fjern parenteser og enheter, senk og underscores)
            rename_clean = {
                "Temperature_2m (°C)": "temperature_2m",
                "Precipitation (mm)": "precipitation",
                "Wind_speed_10m (m/s)": "wind_speed_10m",
                "Wind_gusts_10m (m/s)": "wind_gusts_10m",
                "Wind_direction_10m (°)": "wind_direction_10m",
            }
            for k, v in rename_clean.items():
                if k in df.columns:
                    df = df.rename(columns={k: v})

            # parse date
            if "date" not in df.columns:
                raise KeyError("Finner ikke 'Time' eller 'date' i Open-Meteo-filen.")
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
            return df

    raise FileNotFoundError("Fant ikke open-meteo-subset.csv i 'data/' eller prosjektrot.")


def ensure_openmeteo_in_session():
    if "hourly_dataframe" in st.session_state and not st.session_state["hourly_dataframe"].empty:
        return
    try:
        st.session_state["hourly_dataframe"] = load_openmeteo_csv()
    except Exception as e:
        st.session_state["hourly_dataframe"] = pd.DataFrame()
        st.session_state["__openmeteo_error__"] = str(e)
