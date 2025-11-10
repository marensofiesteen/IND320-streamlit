import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient

st.set_page_config(page_title="Production Overview", layout="wide")
st.title("Yearly & Hourly Production Overview (Elhub 2021)")

try:
    st.header("Yearly & Hourly Production Overview by Price Area and Production Group in 2021")

    # Lazy import + cache for Mongo client
    @st.cache_resource
    def init_connection():
        import pymongo
        # Forventet st.secrets["mongo"]["uri"]
        if "mongo" not in st.secrets or "uri" not in st.secrets["mongo"]:
            return None
        return pymongo.MongoClient(st.secrets["mongo"]["uri"])

    client = init_connection()
    if client is None:
        st.info("Add `st.secrets['mongo']['uri']` to enable this page.")
        st.stop()

    @st.cache_data(ttl=600)
    def get_data():
        db = client["elhub_db"]
        collection = db["production_data"]
        items = list(collection.find())
        return pd.DataFrame(items)

    df = get_data()
    if df.empty:
        st.info("No data found in MongoDB-collection `elhub_db.production_data`.")
        st.stop()

    # Make sure the data types are correct
    # Column names in MongoDB might be lowercase (due to Cassandra), so use safe column naming
    for c in ["starttime", "endtime", "lastupdatedtime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # quantitykwh to numeric
    if "quantitykwh" in df.columns:
        df["quantitykwh"] = pd.to_numeric(df["quantitykwh"], errors="coerce")

    # Layout: two columns
    col1, col2 = st.columns(2)

    # ---- Left: Pie ----
    with col1:
        st.subheader("1. Total Production Distribution by Price Area and Production Group")
        needed = {"pricearea", "productiongroup", "quantitykwh"}
        if not needed.issubset(df.columns):
            st.error(f"Miss columns: {sorted(list(needed - set(df.columns)))}")
        else:
            areas = sorted(df["pricearea"].dropna().unique().tolist())
            if not areas:
                st.info("Did not found any `pricearea`-values.")
            else:
                price_area = st.radio("Step1: Select Price Area", options=areas)
                st.session_state["last_price_area"] = price_area
                df_area = df[df["pricearea"] == price_area]
                df_sub1 = df_area.groupby("productiongroup", dropna=True)["quantitykwh"].sum().reset_index()

                fig1 = px.pie(
                    df_sub1,
                    names="productiongroup",
                    values="quantitykwh",
                    title=f"Total Production - Price Area {price_area}",
                )
                fig1.update_traces(
                    textinfo="percent+label",
                    pull=[0.05] * len(df_sub1),
                    textfont_size=15,
                    domain={"x": [0, 0.9], "y": [0, 0.8]},
                )
                fig1.update_layout(
                    legend_title=dict(text="Production Groups", font=dict(size=18)),
                    margin=dict(t=160),
                    title=dict(text=f"Total Production of each Group - {price_area}", font=dict(size=16), y=0.95),
                    legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
                )
                st.plotly_chart(fig1, width="stretch")

# ---- Right: Lines ----
    with col2:
        st.subheader("2. Hourly Production Trend")

        # 1) Select Production Groups"
        groups_all = sorted(df["productiongroup"].dropna().unique().tolist())
        groups = st.pills(
            "Step2: Select Production Groups",
            options = groups_all,
            selection_mode = "multi",
            default = groups_all,
        )
        if not groups:
            st.info("Select at least one production group.")
            st.stop()

        # 2) Select Month
        month = st.selectbox("Step3: Select Month", list(range(1, 13)))

        # 3) Last selected price area from session state
        price_area_current = st.session_state.get("last_price_area")
        if price_area_current is None and "pricearea" in df.columns and not df["pricearea"].dropna().empty:
            price_area_current = sorted(df["pricearea"].dropna().unique())[0]
        st.session_state["last_price_area"] = price_area_current

        # 4) Filter
        mask = pd.Series(True, index=df.index)
        if price_area_current is not None and "pricearea" in df.columns:
            mask &= df["pricearea"].eq(price_area_current)
        if groups:
            mask &= df["productiongroup"].isin(groups)
        if "starttime" in df.columns:
            mask &= pd.to_datetime(df["starttime"], errors="coerce").dt.month.eq(month)

        df_month = df[mask].copy()

        # 5) Show empty info
        if df_month.empty:
            st.info("No data for seleced.")
        else:
            fig2 = go.Figure()
            for g in groups:
                df_g = df_month[df_month["productiongroup"] == g].sort_values("starttime")
                fig2.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(df_g["starttime"], errors="coerce"),
                            y=df_g["quantitykwh"],
                            mode="lines+markers",
                            name=g,
                        )
                    )
            fig2.update_layout(
                xaxis_title="Date",
                yaxis_title="Quantity kWh",
                title=f"Monthly Production Trend - {price_area_current} Month {month}",
                legend_title="Production Groups",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Data Source"):
        st.write(
            """
            This dataset comes from the Elhub API `PRODUCTION_PER_GROUP_MBA_HOUR`. 
            It was loaded into Cassandra, filtered locally using PySpark, 
            stored in MongoDB, and visualized with Streamlit.
            """
        )
except Exception as e:
    st.error("An error occurred while processing the data.")
    st.exception(e)