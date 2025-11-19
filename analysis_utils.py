"""
Analysis and visualization function without I/O.
- STL decomposition for Elhub production data
- Spectrogram (STFT) for Elhub production data
- SATV + SPC for Open-Meteo
- LOF anomalies for Open-Meteo
Expected columns:
- Elhub functions: ["pricearea", "productiongroup", "datetime", "quantitykwh"]
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
from scipy.fft import dct, idct
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
    period_length: int = 168,       # one week in hours
    seasonal_smoother: int = 7,
    trend_smoother: int = 31 * 24,  # ≈ one month in hours
    robust: bool = True,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    freq: str = "h",
) -> Tuple[go.Figure, STL]:
    
    needed = {"pricearea", "productiongroup", "datetime", "quantitykwh"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    d = df[(df["pricearea"] == price_area) & (df["productiongroup"] == production_group)].copy()
    if start is not None:
        d = d[d["datetime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d[d["datetime"] <= pd.to_datetime(end)]
    if d.empty:
        raise ValueError("No rows after filtering by area/goup/time.")

    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True)
    d = d.sort_values("datetime").set_index("datetime").asfreq(freq)

    y = (
        pd.to_numeric(d["quantitykwh"], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill().bfill()
    )

    seasonal_smoother = _ensure_odd(seasonal_smoother, minimum=7)
    trend_smoother = _ensure_odd(max(trend_smoother, period_length + 1))

    # STL requires some margin relative to the window sizes.
    if len(y) < (period_length + seasonal_smoother + trend_smoother):
        raise ValueError(
            f"Too little data for the selected windows: n={len(y)}, "
            f"Requirement >= {period_length + seasonal_smoother + trend_smoother}"
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


# -------------------- Spectrogram (Elhub-produksjon) --------------------

def spectrogram_production(
    df: pd.DataFrame,
    price_area: str,
    production_group: str,
    window_length_hours: int = 168,   # 1 week
    window_overlap: float = 0.5,      # 50% overlap
    detrend: str = "constant",        # or "linear"
    scaling: str = "psd",             # "density" also works
    clip_db: Optional[Tuple[float, float]] = None,  # None = auto from percentiles
) -> Tuple[go.Figure, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute a time-frequency spectrogram of hourly production (quantitykwh).

    Returns a Plotly Figure and the (frequencies, time_midpoints, Sxx_db) arrays.
    """

    needed = {"pricearea", "productiongroup", "datetime", "quantitykwh"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # 1) Filter and create an hourly time series
    d = (
        df[(df["pricearea"] == price_area) & (df["productiongroup"] == production_group)]
        .dropna(subset=["datetime", "quantitykwh"])
        .sort_values("datetime")
        .copy()
    )
    if d.empty:
        raise ValueError(f"No data for price_area={price_area!r}, production_group={production_group!r}")

    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True)
    d = d.set_index("datetime").asfreq("h")
    y = pd.to_numeric(d["quantitykwh"], errors="coerce").ffill().values  # 1 sample/hour
    t_index = d.index

    # 2) STFT
    fs = 1.0  # samples per hour
    nperseg = int(window_length_hours)
    if nperseg < 8:
        raise ValueError("window_length_hours too small; use at least 8")
    noverlap = int(window_overlap * nperseg)

    f, t_bins, Zxx = stft(
        y,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        return_onesided=True,
        boundary=None,
        padded=False,
        scaling=scaling,
    )

    # Power (magnitude^2) -> dB
    Sxx = np.abs(Zxx) ** 2
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    # 3) Map STFT time bins (center of each window) to real datetimes
    t0 = t_index[0]
    t_mid = pd.to_datetime(t0) + pd.to_timedelta(t_bins, unit="h")

    # 4) Convert frequency (cycles/hour) to period in days for the y-axis
    f_cph = f.copy()
    with np.errstate(divide="ignore"):
        period_days = 24.0 / f_cph
    # Mask very long periods (DC or near-zero freq) for display
    valid = np.isfinite(period_days) & (period_days <= 120)  # hide > ~4 months
    period_days = period_days[valid]
    Sxx_db = Sxx_db[valid, :]

    # 5) Choose a sensible color range
    if clip_db is None:
        # Auto-scale based on percentiles of the actual data
        vmin = float(np.nanpercentile(Sxx_db, 5))
        vmax = float(np.nanpercentile(Sxx_db, 95))
    else:
        vmin, vmax = clip_db

    # 6) Build Plotly heatmap ((x=time, y=period_days, color=power dB)
    fig = go.Figure(
        data=go.Heatmap(
            x=t_mid,
            y=period_days,
            z=Sxx_db,
            colorscale="Viridis",
            colorbar=dict(title="Power (dB)"),
            zsmooth=False,
            zmin=vmin,
            zmax=vmax,
        )
    )
    fig.update_layout(
        title=(
            f"Spectrogram — {price_area} / {production_group} "
            f"(window={window_length_hours}h, overlap={int(window_overlap*100)}%)"
        ),
        xaxis_title="Time",
        yaxis_title="Period (days)",
        template="plotly_white",
        height=600,
    )
    # Shorter periods (higher frequencies) at the top
    fig.update_yaxes(autorange="reversed")

    return fig, (f, t_mid, Sxx_db)


# --------------- SATV + SPC (Open-Meteo, temperature) ---------------

def satv_spc_plot(
    df_hourly: pd.DataFrame,
    temp_col: str = "temperature_2m",
    cutoff: int = 365 * 24 // 12,  # number of DCT coefficients included in the trend
    k_sigma: float = 3.0,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
   Plots RAW temperature data with CURVED SPC boundaries that follow the trend.

    Steps:
    - Creates a smooth trend using DCT (low-pass).
    - High-pass = raw data minus the trend.
    - SPC boundaries are computed on the high-pass component, but added back
    onto the trend → boundaries follow the curve (not straight lines).
    - Outliers = points where the raw data fall outside the curved boundaries.
    """

    # --- Chech columns ---
    if "date" not in df_hourly.columns or temp_col not in df_hourly.columns:
        raise KeyError("Requires columns: 'date' and the selected temperature column.")

    # --- Clean and set hourly frequency ---
    x = df_hourly.copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x = x.sort_values("date").set_index("date").asfreq("h")

    y = (
        pd.to_numeric(x[temp_col], errors="coerce")
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .values
    )

    n = len(y)
    if n < 10:
        raise ValueError(f"Too few samples for SATV/SPC: n={n}")

    cutoff = max(1, min(int(cutoff), n - 1))

    # --- 1) Low-pass trend with DCT ---
    coeff = dct(y, type=2, norm="ortho")
    lp = coeff.copy()
    # low-pass: keep the first ´cutoff´coefficients, zero out the rest
    lp[cutoff:] = 0.0
    trend = idct(lp, type=2, norm="ortho")  # inverse DCT -> smooth trend

    # --- 2) High-pass (deviations from the trend) ---
    hp = y - trend

    # --- 3) Robust SPC on high-pass ---
    med = np.median(hp)
    mad = np.median(np.abs(hp - med)) + 1e-12
    sigma = 1.4826 * mad  # robust std
    lo_hp, hi_hp = med - k_sigma * sigma, med + k_sigma * sigma

    # --- 4) Make the SPC boundaries curved by adding them to the trend ---
    upper_curve = trend + hi_hp
    lower_curve = trend + lo_hp

    # --- 5) Outliers where the RAW data fall outside the curved boundaries ---
    is_out = (y > upper_curve) | (y < lower_curve)
    t = x.index

    out = (
        pd.DataFrame(
            {
                "date": t,
                temp_col: y,
                "trend": trend,
                "upper_curve": upper_curve,
                "lower_curve": lower_curve,
                "is_outlier": is_out,
            }
        )
        .loc[is_out]
        .assign(
            direction=lambda df_: np.where(
                df_[temp_col] > df_["trend"], "high", "low"
            )
        )
    )

    # --- 6) Plotly figures: raw data + trend + curved boundaries + outliers ---
    fig = go.Figure()

    # Rådata
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="lines",
            name="Temperature (raw)",
        )
    )

    # Trend
    fig.add_trace(
        go.Scatter(
            x=t,
            y=trend,
            mode="lines",
            name="Trend (low-pass)",
            line=dict(dash="dot"),
        )
    )

    # Upper and lower curved SPC boundaries
    fig.add_trace(
        go.Scatter(
            x=t,
            y=upper_curve,
            mode="lines",
            name=f"Upper SPC ({k_sigma}σ, curved)",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=lower_curve,
            mode="lines",
            name=f"Lower SPC ({k_sigma}σ, curved)",
            line=dict(dash="dash"),
        )
    )

    # Outliers as points on the raw data
    if is_out.any():
        fig.add_trace(
            go.Scatter(
                x=t[is_out],
                y=y[is_out],
                mode="markers",
                name="Outliers",
                marker=dict(symbol="x", size=8),
            )
        )

    fig.update_layout(
        height=600,
        template="plotly_white",
        yaxis_title=temp_col,
        xaxis_title="Time",
        title=f"Raw {temp_col} with curved SPC boundaries (cutoff={cutoff}, k_sigma={k_sigma})",
    )

    return fig, out


# ---------------------- LOF-anomalies (Open-Meteo) ----------------------

def lof_anomaly_plot(
    df_hourly: pd.DataFrame,
    var_col: str = "precipitation",
    contamination: float = 0.01,
    n_neighbors: int = 20,
) -> Tuple[go.Figure, pd.DataFrame]:
    
    if "date" not in df_hourly.columns or var_col not in df_hourly.columns:
        raise KeyError("Requires columns: 'date' and the selected variable column.")

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
    fig.add_trace(go.Scatter(x=t[is_out], y=v[is_out].ravel(), mode="markers", name="Anomalies"))
    fig.update_layout(height=500, template="plotly_white", yaxis_title=var_col)

    out = pd.DataFrame({"date": t[is_out], var_col: v[is_out].ravel()})
    return fig, out


