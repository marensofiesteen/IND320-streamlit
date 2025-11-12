import requests, pandas as pd, streamlit as st

PRICE_AREAS = {
    "NO1": {"city": "Oslo",        "lat": 59.9139, "lon": 10.7522},
    "NO2": {"city": "Kristiansand","lat": 58.1467, "lon": 7.9956},
    "NO3": {"city": "Trondheim",   "lat": 63.4305, "lon": 10.3951},
    "NO4": {"city": "Tromsø",      "lat": 69.6492, "lon": 18.9560},
    "NO5": {"city": "Bergen",      "lat": 60.39299,"lon": 5.32415},
}

@st.cache_data(show_spinner=False)
def fetch_openmeteo_hourly(price_area: str, year: int = 2021) -> pd.DataFrame:
    meta = PRICE_AREAS[price_area]
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = dict(
        latitude=meta["lat"], longitude=meta["lon"],
        start_date=f"{year}-01-01", end_date=f"{year}-12-31",
        hourly="temperature_2m,precipitation,wind_speed_10m",
        timezone="Europe/Oslo",
    )
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    hourly = r.json().get("hourly", {})
    if "time" not in hourly: 
        return pd.DataFrame()
    df = pd.DataFrame(hourly).rename(columns={"time":"date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["priceArea"] = price_area
    df["city"] = meta["city"]
    return df
