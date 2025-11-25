import json
from datetime import date

import folium
import pandas as pd
import streamlit as st
from pymongo import MongoClient, errors as mongo_errors
from streamlit_folium import st_folium
import branca.colormap as cm

from utils.navigation import sidebar_navigation

# Local meny for this group
sidebar_navigation("Snow & geo")

st.set_page_config(page_title="Elhub map – price areas and municipalities",
                   layout="wide")
st.title("Price areas NO1–NO5 and municipalities – map view")

# ---------------------------------------------------------
#  Config
# ---------------------------------------------------------
MUNICIPALITY_ZOOM_THRESHOLD = 6  # show municipalities when zoomed in further than this


# ---------------------------------------------------------
#  Helpers for price areas
# ---------------------------------------------------------
def get_price_area_code(feature):
    """
    Extract the price area code (NO1–NO5) from the GeoJSON feature properties.
    Tries several possible keys and falls back to any value that looks like 'NO1'...'NO5'.
    """
    props = feature.get("properties", {})

    candidates = [
        props.get("ElSpotOmr"),
        props.get("pricearea"),
        props.get("PriceArea"),
    ]

    for c in candidates:
        if isinstance(c, str):
            code = c.strip().upper()
            if code.startswith("NO") and len(code) == 3:
                return code

    # Fallback: scan all property values
    for v in props.values():
        if isinstance(v, str):
            code = v.strip().upper()
            if code.startswith("NO") and len(code) == 3:
                return code

    return None


@st.cache_resource(show_spinner=True)
def load_price_area_geojson(path: str = "data/elspot_areas.geojson"):
    """
    Load the GeoJSON file with Elspot price areas (NO1–NO5).
    The file is downloaded from NVE and stored locally in the data/ folder.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
    except FileNotFoundError:
        st.error(f"GeoJSON file not found at '{path}'. Please check the path.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Could not decode GeoJSON file '{path}': {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while loading GeoJSON '{path}': {e}")
        return None

    return gj


# ---------------------------------------------------------
#  Load GeoJSON for municipalities (cached)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_municipality_geojson(path: str = "data/municipalities.geojson"):
    """
    Load the GeoJSON file with Norwegian municipalities.
    Handles BOM by using utf-8-sig and adds:
      - a simple 'id' field per feature
      - a 'label_name' property for tooltips (municipality name)
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            gj = json.load(f)
    except FileNotFoundError:
        st.error(f"Municipality GeoJSON file not found at '{path}'. Please check the path.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Could not decode municipality GeoJSON file '{path}': {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while loading municipality GeoJSON '{path}': {e}")
        return None

    feats = gj.get("features", [])

    for i, feat in enumerate(feats):
        # Add simple numeric id so Folium does NOT use nested property keys as identifier
        feat["id"] = str(i)

        # Extract municipality name from any of the possible fields
        props = feat.get("properties", {})
        muni_name = (
            props.get("kommunenavn")
            or props.get("KOMMUNENAVN")
            or props.get("navn")
            or props.get("NAVN")
            or "Unknown municipality"
        )

        # Add a unified property for tooltip display
        props["label_name"] = muni_name

    return gj


# ---------------------------------------------------------
#  MongoDB loader for production/consumption data (cached)
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_elhub_collection(collection_name: str) -> pd.DataFrame:
    """
    Load a full MongoDB collection (production or consumption) into a pandas DataFrame.
    Uses connection details from Streamlit secrets.
    Includes simple error handling and basic datetime parsing.
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
        client.server_info()
        db = client[db_name]
        coll = db[collection_name]
        docs = list(coll.find({}, {"_id": 0}))
    except mongo_errors.ServerSelectionTimeoutError:
        st.error("Could not reach MongoDB server. Check network or MongoDB settings.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error while loading data from MongoDB collection '{collection_name}': {e}")
        return pd.DataFrame()

    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df.columns = [c.lower() for c in df.columns]

    # Parse datetime columns
    for col in ["starttime", "endtime", "lastupdatedtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop invalid starttime (needed for date filtering)
    if "starttime" in df.columns:
        df = df.dropna(subset=["starttime"])

    # Drop missing quantity if present
    if "quantitykwh" in df.columns:
        df = df.dropna(subset=["quantitykwh"])

    return df


# ---------------------------------------------------------
#  Load data
# ---------------------------------------------------------
with st.spinner("Loading GeoJSON and Elhub data from MongoDB ..."):
    geojson_price = load_price_area_geojson()
    geojson_muni = load_municipality_geojson()
    df_prod = load_elhub_collection("production_data")
    df_cons = load_elhub_collection("consumption_data")

if geojson_price is None or geojson_muni is None:
    st.stop()

if df_prod.empty and df_cons.empty:
    st.error("No production or consumption data found in MongoDB. Map cannot be computed.")
    st.stop()

if df_prod.empty:
    st.error("No production data found in MongoDB.")
    st.stop()

# Read last selected coordinate and zoom from session_state (if any)
selected_coord = st.session_state.get("selected_coord")  # {"lat": ..., "lon": ...}
current_zoom = st.session_state.get("map_zoom", 4)


# ---------------------------------------------------------
#  Sidebar controls: data type, group, time interval
# ---------------------------------------------------------
st.sidebar.header("Selections")

data_type = st.sidebar.radio(
    "Data type",
    ["Production", "Consumption"],
    horizontal=True,
)

if data_type == "Production":
    df_all = df_prod.copy()
    group_col = "productiongroup"
    title_prefix = "Production"
else:
    if df_cons.empty:
        st.error("Consumption data is not available in MongoDB.")
        st.stop()
    df_all = df_cons.copy()
    group_col = "consumptiongroup"
    title_prefix = "Consumption"

required_cols = {"pricearea", group_col, "starttime", "quantitykwh"}
missing = required_cols - set(df_all.columns)
if missing:
    st.error(f"Missing expected columns in {data_type.lower()} data: {missing}")
    st.stop()

# Derive date for filtering
df_all["date"] = df_all["starttime"].dt.date

if df_all["date"].isna().all():
    st.error("All 'starttime' values are invalid; cannot derive dates for filtering.")
    st.stop()

min_date = df_all["date"].min()
max_date = df_all["date"].max()

if not isinstance(min_date, date) or not isinstance(max_date, date):
    st.error("Could not determine valid min/max dates from the data.")
    st.stop()

st.sidebar.write("Time interval (days)")
date_range = st.sidebar.date_input(
    "Select start and end date",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    # If user only selects one date, use it as both start and end
    start_date = end_date = date_range

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Filter by date interval
mask_time = (df_all["date"] >= start_date) & (df_all["date"] <= end_date)
df_filtered = df_all.loc[mask_time].copy()

if df_filtered.empty:
    st.warning("No data for the selected time interval.")
    st.stop()

# Available groups
available_groups = sorted(df_filtered[group_col].dropna().unique())
default_groups = st.sidebar.multiselect(
    "Groups",
    options=available_groups,
    default=available_groups,
)

if not default_groups:
    st.sidebar.error("Please select at least one group.")
    st.stop()

mask_group = df_filtered[group_col].isin(default_groups)
df_filtered = df_filtered.loc[mask_group].copy()

if df_filtered.empty:
    st.warning("No data for the selected groups and time interval.")
    st.stop()

st.sidebar.markdown(
    f"**{title_prefix}** aggregated as mean `quantitykwh` "
    f"per price area over the selected period."
)

# ---------------------------------------------------------
#  Aggregate to mean per price area
# ---------------------------------------------------------
agg = (
    df_filtered.groupby("pricearea")["quantitykwh"]
    .mean()
    .reset_index()
    .rename(columns={"quantitykwh": "mean_quantity_kwh"})
)

# Also compute GWh for nicer numbers in the legend
agg["mean_quantity_gwh"] = agg["mean_quantity_kwh"] / 1e6

st.subheader(f"{title_prefix}: mean energy per price area")
st.write(
    f"Time interval: {start_date} → {end_date} "
    f"({(end_date - start_date).days + 1} days)"
)
st.dataframe(agg, use_container_width=True)

# Mapping pricearea -> mean_quantity in GWh for colouring
area_to_value = {
    str(row["pricearea"]).strip().upper(): row["mean_quantity_gwh"]
    for _, row in agg.iterrows()
}

if not area_to_value:
    st.warning("No aggregated values available.")
    st.stop()

vmin = min(area_to_value.values())
vmax = max(area_to_value.values())

# ---------------------------------------------------------
#  Color scale (use step colormap to avoid overlapping labels)
# ---------------------------------------------------------
colormap = cm.linear.YlGnBu_09.scale(vmin, vmax).to_step(5)
colormap.caption = "Mean quantity [GWh]"


# ---------------------------------------------------------
#  Helper functions for styling and properties
# ---------------------------------------------------------
def style_function_price(feature):
    """
    Base styling for price area polygons, using the aggregated mean values.
    """
    area = get_price_area_code(feature)
    value = area_to_value.get(area)

    if value is None:
        # No data for this area
        return {
            "fillColor": "#cccccc",
            "color": "black",
            "weight": 2,
            "fillOpacity": 0.3,
        }

    return {
        "fillColor": colormap(value),
        "color": "black",
        "weight": 2,
        "fillOpacity": 0.7,
    }


def highlight_function(_feature):
    """
    Styling when hovering a polygon.
    """
    return {
        "fillColor": "#ffcc00",
        "color": "red",
        "weight": 2,
        "fillOpacity": 0.7,
    }


def style_function_muni(_feature):
    """
    Styling for municipality polygons.
    We keep them transparent with a thin border so they work well on top of price areas.
    """
    return {
        "fillColor": "#00000000",  # transparent fill
        "color": "#555555",
        "weight": 0.7,
        "fillOpacity": 0.0,
    }


def get_municipality_name(feature):
    """
    Try to extract municipality name from different possible property keys.
    """
    props = feature.get("properties", {})
    return (
        props.get("KOMMUNENAVN")
        or props.get("kommunenavn")
        or props.get("NAVN")
        or props.get("navn")
    )


# ---------------------------------------------------------
#  Build Folium map
# ---------------------------------------------------------
m = folium.Map(location=[65.0, 15.0], zoom_start=current_zoom)

# Price areas (always visible)
folium.GeoJson(
    geojson_price,
    name="Elspot price areas",
    style_function=style_function_price,
    highlight_function=highlight_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["ElSpotOmr"],   # <-- direkte fra NVE GeoJSON
        aliases=["Price area:"],
        labels=True,
        sticky=False,
    ),
).add_to(m)

# Municipalities (only visible when zoomed in enough)
if current_zoom >= MUNICIPALITY_ZOOM_THRESHOLD:
    folium.GeoJson(
        geojson_muni,
        name="Municipalities",
        style_function=style_function_muni,
        tooltip=folium.GeoJsonTooltip(
            fields=["label_name"],
            aliases=["Municipality"],
            labels=True,
            sticky=False,
        ),
    ).add_to(m)

# Add color scale legend
colormap.add_to(m)

# If we already have a selected coordinate, show it as a small marker
if selected_coord is not None:
    folium.CircleMarker(
        location=[selected_coord["lat"], selected_coord["lon"]],
        radius=6,
        color="red",
        fill=True,
        fill_opacity=0.9,
    ).add_to(m)

st.write(
    "Zoom out to see price areas NO1–NO5. "
    f"When you zoom in beyond level {MUNICIPALITY_ZOOM_THRESHOLD}, "
    "municipality borders are shown on top. "
    "Click on the map to select a coordinate."
)

map_data = st_folium(m, width=900, height=600, key="elhub_map")

# Update selected coordinate if user clicked
if map_data is not None:
    # Zoom handling
    new_zoom = map_data.get("zoom")
    if isinstance(new_zoom, (int, float)):
        st.session_state["map_zoom"] = new_zoom

    # Click handling
    if map_data.get("last_clicked") is not None:
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]

        st.session_state["selected_coord"] = {"lat": lat, "lon": lon}
        selected_coord = st.session_state["selected_coord"]

        st.success(f"Selected coordinate (this session): {lat:.4f}, {lon:.4f}")
    else:
        selected_coord = st.session_state.get("selected_coord")

        if selected_coord is not None:
            st.info(
                f"Current stored coordinate (this session): "
                f"{selected_coord['lat']:.4f}, {selected_coord['lon']:.4f}"
            )
        else:
            st.info("No coordinate selected yet in this session.")
else:
    if selected_coord is not None:
        st.info(
            f"Current stored coordinate (this session): "
            f"{selected_coord['lat']:.4f}, {selected_coord['lon']:.4f}"
        )
    else:
        st.info("No coordinate selected yet in this session.")

# Show current zoom level as a reference
zoom_for_display = st.session_state.get("map_zoom", current_zoom)
st.caption(
    f"Current zoom level: **{zoom_for_display:.1f}** "
    f"(municipalities are shown from **{MUNICIPALITY_ZOOM_THRESHOLD}** and up)."
)


with st.expander("About this page"):
    st.markdown(
        f"""
        - The polygons are:
          - **Elspot price areas (NO1–NO5)** from NVE (GeoJSON).
          - **Municipality borders** from Geonorge
            (EUREF 89 Geographical, ETRS89 2D, GeoJSON).
        - The map colours each price area according to the **mean quantity**,
          aggregated over the selected time interval and groups.
          The legend shows values in **GWh**.
        - When zoomed out, you mainly see price areas.  
          When you zoom in beyond level **{MUNICIPALITY_ZOOM_THRESHOLD}**, 
          municipality borders are added on top.
        - Data comes from MongoDB Atlas collections:
          `production_data` and `consumption_data`.
        - When you click on the map, the coordinate is stored in
          `st.session_state["selected_coord"]` and can be used on other pages
          (e.g., snow drift and correlation analysis).
        """
    )
