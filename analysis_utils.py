"""
Analyse- og visualiseringsfunksjoner uten I/O.
- STL-dekomponering for Elhub-produksjonsdata
- Spektrogram (STFT) for Elhub-produksjonsdata
- SATV + SPC for Open-Meteo
- LOF-anomalier for Open-Meteo

Forventede kolonner:
- Elhub-funksjoner: ["pricearea", "productiongroup", "datetime", "quantitykwh"]
- Open-Meteo-funksjoner: ["date", <valgt variabel>]

I/O (lasting av CSV, normalisering av kolonner, session_state) skal håndteres
utenfor dette modulen (f.eks. i utils/io_utils.py).
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from scipy.fft import dct
from sklearn.neighbors import LocalOutlierFactor
from scipy.signal import stft

# -------- STL (Elhub) --------
def _ensure_odd(n: int, minimum: int = 3) -> int:
    # Ensure window length is odd and >= minimum
    n = int(n)
    if n < minimum: n = minimum
    if n % 2 == 0: n += 1
    return n

def stl_decompose_production(
    df: pd.DataFrame,
    price_area: str,
    production_group: str,
    period_length: int = 168,       # én uke i timer
    seasonal_smoother: int = 7,
    trend_smoother: int = 31 * 24,  # ≈ én måned i timer
    robust: bool = True,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    freq: str = "h",
) -> Tuple[go.Figure, STL]:
    """
    STL-dekomponering av timeoppløst produksjonsdata.

    Parametre:
        df: DataFrame med kolonner {"pricearea","productiongroup","datetime","quantitykwh"}.
        price_area, production_group: Filtre.
        period_length: Sesongperiode i time-trinn (168 = ukesyklus).
        seasonal_smoother, trend_smoother: LOESS-vinduer (må være oddetall).
        robust: Robust fit i STL.
        start, end: Valgfritt tidsfilter.
        freq: Frekvens for re-sampling (default "h" = hourly).

    Returnerer:
        (plotly.figure, STL-resultat-objekt)
    """
    needed = {"pricearea", "productiongroup", "datetime", "quantitykwh"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise KeyError(f"Mangler kolonner: {missing}")

    d = df[(df["pricearea"] == price_area) & (df["productiongroup"] == production_group)].copy()
    if start is not None:
        d = d[d["datetime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d[d["datetime"] <= pd.to_datetime(end)]
    if d.empty:
        raise ValueError("Ingen rader etter filtrering på area/group/tid.")

    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True)
    d = d.sort_values("datetime").set_index("datetime").asfreq(freq)

    y = (
        pd.to_numeric(d["quantitykwh"], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill().bfill()
    )

    seasonal_smoother = _ensure_odd(seasonal_smoother, minimum=7)
    trend_smoother = _ensure_odd(max(trend_smoother, period_length + 1))

    # STL trenger litt margin ift. vinduer
    if len(y) < (period_length + seasonal_smoother + trend_smoother):
        raise ValueError(
            f"For lite data for vinduer valgt: n={len(y)}, "
            f"krav >= {period_length + seasonal_smoother + trend_smoother}"
        )

    stl = STL(y, period=period_length, seasonal=seasonal_smoother, trend=trend_smoother, robust=robust).fit()

    # Plot
    t = y.index
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=("Original", "Trend", "Seasonal", "Remainder")
    )
    fig.add_trace(go.Scatter(x=t, y=y, name="Original", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=stl.trend, name="Trend", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=stl.seasonal, name="Seasonal", mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=stl.resid, name="Remainder", mode="lines"), row=4, col=1)
    fig.update_layout(height=900, template="plotly_white", margin=dict(t=80))

    return fig, stl


# -------------------- Spektrogram (Elhub-produksjon) --------------------

def spectrogram_production(
    df: pd.DataFrame,
    price_area: str,
    production_group: str,
    window_length_hours: int = 168,
    window_overlap: float = 0.5,
    fs: float = 1.0,                 # 1 sample per time
    clip_db: Tuple[float, float] = (-60, 20),
) -> Tuple[go.Figure, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    STFT-basert spektrogram av timeoppløst produksjon.

    Returnerer:
        (plotly.figure, (frekvenser, tids-bin, spekter[dB]))
    """
    needed = {"pricearea", "productiongroup", "datetime", "quantitykwh"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise KeyError(f"Mangler kolonner: {missing}")

    d = df[(df["pricearea"] == price_area) & (df["productiongroup"] == production_group)].copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True)
    d = d.sort_values("datetime").set_index("datetime").asfreq("h")

    y = (
        pd.to_numeric(d["quantitykwh"], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill().bfill()
        .values
    )

    nperseg = int(window_length_hours)
    if nperseg < 8:
        raise ValueError("window_length_hours for liten; bruk minst 8.")
    if len(y) < nperseg:
        raise ValueError(f"Tidsserie for kort ({len(y)}) for valgt vindu {nperseg}.")

    noverlap = int(max(0.0, min(0.95, float(window_overlap))) * nperseg)

    f, t_bins, Zxx = stft(
        y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        detrend=False, boundary=None, padded=False
    )
    Sxx = np.abs(Zxx) ** 2
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    vmin, vmax = clip_db
    fig = go.Figure(data=go.Heatmap(
        x=t_bins, y=f, z=Sxx_db, colorscale="Viridis", zmin=vmin, zmax=vmax,
        colorbar=dict(title="dB")
    ))
    fig.update_layout(
        height=600, template="plotly_white",
        xaxis_title="Tids-bin (timer)", yaxis_title="Frekvens (sykluser/time)"
    )
    return fig, (f, t_bins, Sxx_db)


# --------------- SATV + SPC (Open-Meteo, temperatur) ---------------

def satv_spc_plot(
    df_hourly: pd.DataFrame,
    temp_col: str = "temperature_2m",
    cutoff: int = 365 * 24 // 12,  # fjern lavfrekvente komponenter
    k_sigma: float = 3.0,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Seasonally Adjusted Temperature Variations via high-pass DCT + SPC-bånd.
    Forventer kolonner: ["date", temp_col].

    Returnerer:
        (plotly.figure, outliers_df med kolonner ["date", "satv"])
    """
    if "date" not in df_hourly.columns or temp_col not in df_hourly.columns:
        raise KeyError("Trenger kolonner: 'date' og valgt temperaturkolonne.")

    x = df_hourly.copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x = x.sort_values("date").set_index("date").asfreq("h")

    y = (
        pd.to_numeric(x[temp_col], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill().bfill()
        .values
    )

    # DCT high-pass: null ut lavfrekvente komponenter (<= cutoff)
    coeff = dct(y, type=2, norm="ortho")
    hp = coeff.copy()
    cutoff = max(1, int(cutoff))
    hp[:cutoff] = 0.0
    satv = dct(hp, type=3, norm="ortho")  # invers-approks

    med = np.median(satv)
    mad = np.median(np.abs(satv - med)) + 1e-12
    sigma = 1.4826 * mad
    lo, hi = med - k_sigma * sigma, med + k_sigma * sigma

    t = x.index
    is_out = (satv < lo) | (satv > hi)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=satv, mode="lines", name="SATV"))
    fig.add_trace(go.Scatter(x=t, y=np.full_like(satv, lo), name=f"Nedre ({k_sigma}σ)", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=np.full_like(satv, hi), name=f"Øvre ({k_sigma}σ)", mode="lines"))
    fig.add_trace(go.Scatter(x=t[is_out], y=np.array(satv)[is_out], mode="markers", name="Avvik"))
    fig.update_layout(height=500, template="plotly_white", yaxis_title="SATV")

    out = pd.DataFrame({"date": t[is_out], "satv": np.array(satv)[is_out]})
    return fig, out


# ---------------------- LOF-anomalier (Open-Meteo) ----------------------

def lof_anomaly_plot(
    df_hourly: pd.DataFrame,
    var_col: str = "precipitation",
    contamination: float = 0.01,
    n_neighbors: int = 20,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Local Outlier Factor på én timet variabel.
    Forventer kolonner: ["date", var_col].

    Returnerer:
        (plotly.figure, anomalies_df med kolonner ["date", var_col])
    """
    if "date" not in df_hourly.columns or var_col not in df_hourly.columns:
        raise KeyError("Trenger kolonner: 'date' og valgt variabelkolonne.")

    x = df_hourly.copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x = x.sort_values("date").set_index("date").asfreq("h")
    v = pd.to_numeric(x[var_col], errors="coerce").fillna(0.0).values.reshape(-1, 1)

    lof = LocalOutlierFactor(n_neighbors=int(n_neighbors), contamination=float(contamination))
    y_pred = lof.fit_predict(v)  # -1 = outlier
    is_out = (y_pred == -1)

    t = x.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=v.ravel(), mode="lines", name=var_col))
    fig.add_trace(go.Scatter(x=t[is_out], y=v[is_out].ravel(), mode="markers", name="Anomalier"))
    fig.update_layout(height=500, template="plotly_white", yaxis_title=var_col)

    out = pd.DataFrame({"date": t[is_out], var_col: v[is_out].ravel()})
    return fig, out


