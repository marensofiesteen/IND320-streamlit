import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient, errors as mongo_errors

from utils.utils_openmeteo import PRICE_AREAS
from utils.navigation import sidebar_navigation

# Local meny for this group
sidebar_navigation("Explorative / overview")

st.set_page_config(page_title="Energy overview – Elhub", layout="wide")

# ---------------------------------------------------------
#  MongoDB loader with basic error handling
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_elhub_collection(collection_name: str) -> pd.DataFrame:
    """
    Load a full MongoDB collection (production or consumption) into a pandas DataFrame.
    Uses connection details from Streamlit secrets.
    Includes simple error handling for connection issues and invalid secrets.
    """
    try:
        mongo_secrets = st.secrets["mongo"]
        uri = mongo_secrets["uri"]
        db_name = mongo_secrets["db"]
    except Exception as e:
        st.error(f"Missing or invalid MongoDB configuration in secrets: {e}")
        return pd.DataFrame()

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Trigger a connection test
        client.server_info()
        db = client[db_name]
        coll = db[collection_name]
        docs = list(coll.find({}, {"_id": 0}))  # Exclude _id to keep DataFrame clean
    except mongo_errors.ServerSelectionTimeoutError:
        st.error("Could not reach MongoDB server. Check network or MongoDB settings.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error while loading data from MongoDB collection '{collection_name}': {e}")
        return pd.DataFrame()

    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)

    # Normalise column names to lowercase for consistency
    df.columns = [c.lower() for c in df.columns]

    # Try to parse datetime columns if present
    for col in ["starttime", "endtime", "lastupdatedtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop rows with invalid starttime (critical for time-based filtering)
    if "starttime" in df.columns:
        df = df.dropna(subset=["starttime"])

    return df


# ---------------------------------------------------------
#  Price area list from PRICE_AREAS
# ---------------------------------------------------------
if isinstance(PRICE_AREAS, dict):
    PRICE_AREA_LIST = list(PRICE_AREAS.keys())
else:
    PRICE_AREA_LIST = list(PRICE_AREAS)

PRICE_AREA_LIST = sorted(PRICE_AREA_LIST)


# ---------------------------------------------------------
#  Load production and consumption data
# ---------------------------------------------------------
with st.spinner("Loading Elhub production and consumption data from MongoDB ..."):
    df_prod = load_elhub_collection("production_data")
    df_cons = load_elhub_collection("consumption_data")

if df_prod.empty and df_cons.empty:
    st.error("No production or consumption data found in MongoDB. Please check your database and data pipeline.")
    st.stop()

if df_prod.empty:
    st.error("No production data found in MongoDB. This page requires production data.")
    st.stop()

if df_cons.empty:
    st.warning("No consumption data found in MongoDB. Only production will be available.")


# ---------------------------------------------------------
#  Read global defaults from session_state
# ---------------------------------------------------------
default_area = st.session_state.get("price_area", "NO1")
if default_area not in PRICE_AREA_LIST:
    default_area = PRICE_AREA_LIST[0]

# Prefer 'selected_year' (from Home), fall back to 'year' or 2021
default_year = int(
    st.session_state.get(
        "selected_year",
        st.session_state.get("year", 2021),
    )
)

default_month = int(st.session_state.get("month", 1))
default_data_type = st.session_state.get("data_type", "Production")


# ---------------------------------------------------------
#  Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Selections")

data_type = st.sidebar.radio(
    "Data type",
    options=["Production", "Consumption"],
    index=0 if default_data_type == "Production" else 1,
    horizontal=True,
)
st.session_state["data_type"] = data_type

# If user selects Consumption but we have no data, stop with a helpful error
if data_type == "Consumption" and df_cons.empty:
    st.error(
        "Consumption data is not available in MongoDB. "
        "Please choose 'Production' or load the consumption dataset."
    )
    st.stop()

price_area = st.sidebar.selectbox(
    "Price area",
    options=PRICE_AREA_LIST,
    index=PRICE_AREA_LIST.index(default_area),
)
st.session_state["price_area"] = price_area

# Choose which DataFrame to use based on data type
if data_type == "Production":
    df_all = df_prod.copy()
    group_col = "productiongroup"
    title_prefix = "Production"
else:
    df_all = df_cons.copy()
    group_col = "consumptiongroup"
    title_prefix = "Consumption"

# ---------------------------------------------------------
#  Ensure required columns exist
# ---------------------------------------------------------
required_cols = {"pricearea", group_col, "starttime", "quantitykwh"}
missing = required_cols - set(df_all.columns)
if missing:
    st.error(f"Missing expected columns in {data_type.lower()} data: {missing}")
    st.stop()

# Drop rows with missing quantitykwh (avoid NaNs in aggregations/plots)
if "quantitykwh" in df_all.columns:
    df_all = df_all.dropna(subset=["quantitykwh"])

# ---------------------------------------------------------
#  Derive year and month for filtering
# ---------------------------------------------------------
df_all["year"] = df_all["starttime"].dt.year
df_all["month"] = df_all["starttime"].dt.month

available_years = sorted(df_all["year"].unique())
if not available_years:
    st.error("No valid years found in the data.")
    st.stop()

if default_year not in available_years:
    default_year = available_years[0]

year = st.sidebar.selectbox(
    "Year",
    options=available_years,
    index=available_years.index(default_year),
)
# Keep both keys in session_state for compatibility
st.session_state["year"] = int(year)
st.session_state["selected_year"] = int(year)

# Clamp default_month to valid range [1, 12]
default_month = max(1, min(12, default_month))

month = st.sidebar.selectbox(
    "Month",
    options=list(range(1, 13)),
    index=default_month - 1,
)
st.session_state["month"] = int(month)

# Determine available groups for the current data type
available_groups = sorted(df_all[group_col].dropna().unique())
default_groups_all = st.session_state.get("groups", available_groups)

# Sanitise defaults (ensure all are in available_groups)
default_groups = [g for g in default_groups_all if g in available_groups]
if not default_groups:
    default_groups = available_groups


# ---------------------------------------------------------
#  Main layout
# ---------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Total distribution (selected month)")
    st.caption(f"{title_prefix} per group, {price_area}, {month:02d}/{year}")

with col_right:
    st.subheader("2. Hourly trend (selected month)")
    st.caption(f"{title_prefix} per group over time")


# ---------------------------------------------------------
#  Filter DataFrame for selected area/year/month
# ---------------------------------------------------------
mask_base = (
    (df_all["pricearea"] == price_area)
    & (df_all["year"] == year)
    & (df_all["month"] == month)
)

df_filtered = df_all.loc[mask_base].copy()

if df_filtered.empty:
    st.info("No data for the selected combination of area / year / month.")
    st.stop()


# ---------------------------------------------------------
#  Left column: pie chart (total per group in month)
# ---------------------------------------------------------
with col_left:
    pie_df = (
        df_filtered.groupby(group_col)["quantitykwh"]
        .sum()
        .reset_index()
        .sort_values("quantitykwh", ascending=False)
    )

    if pie_df.empty:
        st.info("No data available for the pie chart.")
    else:
        fig_pie = px.pie(
            pie_df,
            names=group_col,
            values="quantitykwh",
            title=f"{title_prefix} per group – {price_area}, {month:02d}/{year}",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------
#  Right column: line chart with group selection
# ---------------------------------------------------------
with col_right:
    sel_groups = st.pills(
        "Select groups",
        options=available_groups,
        selection_mode="multi",
        default=default_groups,
    )
    st.session_state["groups"] = sel_groups

    df_line = df_filtered[df_filtered[group_col].isin(sel_groups)]

    if df_line.empty:
        st.info("No data for the selected groups.")
    else:
        # Sort by time and plot as multi-line using Plotly
        df_line = df_line.sort_values("starttime")

        fig_line = px.line(
            df_line,
            x="starttime",
            y="quantitykwh",
            color=group_col,
            labels={
                "starttime": "Time",
                "quantitykwh": "Energy [kWh]",
                group_col: "Group",
            },
            title=f"{title_prefix} – hourly values, {price_area}, {month:02d}/{year}",
        )
        fig_line.update_layout(hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)


# ---------------------------------------------------------
#  Documentation / pipeline explanation
# ---------------------------------------------------------
with st.expander("Data source and processing pipeline"):
    st.markdown(
        """
**Data pipeline**

- Elhub API  
  - `PRODUCTION_PER_GROUP_MBA_HOUR` (production, 2021–2024)  
  - `CONSUMPTION_PER_GROUP_MBA_HOUR` (consumption, 2021–2024)  
- Local Cassandra (raw hourly data per price area and production/consumption group)  
- Spark → export to MongoDB Atlas collections:
  - `production_data`
  - `consumption_data`
- This page reads directly from MongoDB using credentials stored in `secrets.toml`.

Selections are shared via `st.session_state` (price area, year, month, groups, data type)
and can be reused on other pages in the app for consistent analysis.
"""
    )
