import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from db_mongo import fetch_month
from utils.utils_openmeteo import fetch_openmeteo_hourly, PRICE_AREAS

st.set_page_config(page_title="Production Elhub", layout="wide")
st.title("Yearly & Hourly Production Overview (Elhub 2021)")

price_areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
groups_all = ["hydro", "wind", "thermal", "solar", "other"]

# --- Read global defaults from session_state ---
default_area = st.session_state.get("price_area", "NO1")
if default_area not in price_areas:
    default_area = "NO1"

default_month = int(st.session_state.get("month", 1))
if default_month < 1 or default_month > 12:
    default_month = 1

default_groups = st.session_state.get("groups", groups_all)

col1, col2 = st.columns(2)
with col1:
    price_area = st.radio(
        "Step 1: Select price area",
        options=price_areas,
        index=price_areas.index(default_area),
        horizontal=True,
    )
    # Keep global selection in sync
    st.session_state["price_area"] = price_area

with col2:
    month = st.selectbox(
        "Step 2: Select month",
        list(range(1, 13)),
        index=default_month - 1,
    )
    st.session_state["month"] = int(month)

# Left side: pie chart (sum per group in the selected month)
df_month_all = fetch_month(price_area, groups_all, int(month))

left, right = st.columns(2)

with left:
    st.subheader("1. Total production distribution (selected month)")
    if df_month_all.empty:
        st.info("No data for the selected area/month.")
    else:
        pie_df = (
            df_month_all.groupby("productiongroup")["quantitykwh"]
            .sum()
            .reset_index()
        )
        fig1 = px.pie(
            pie_df,
            names="productiongroup",
            values="quantitykwh",
            title=f"Total per group – {price_area}, {month:02d}/2021",
        )
        st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("2. Hourly production trend (selected month)")

    sel_groups = st.pills(
        "Select production groups",
        options=groups_all,
        selection_mode="multi",
        default=default_groups,
    )
    # Sync global selection of groups
    st.session_state["groups"] = sel_groups

    df = fetch_month(price_area, sel_groups, int(month))
    if df.empty:
        st.info("No data for the selected combination.")
    else:
        pivot = (
            df.pivot_table(
                index="datetime",
                columns="productiongroup",
                values="quantitykwh",
            )
            .sort_index()
        )
        st.line_chart(pivot, use_container_width=True)

with st.expander("Data source"):
    st.write(
        """
Elhub API `PRODUCTION_PER_GROUP_MBA_HOUR` (2021) → Cassandra (raw) → Spark filtering → MongoDB (curated).
This page reads directly from MongoDB via secrets, and uses the global
price area, month, and production-group selections defined on the Home page.
"""
    )

