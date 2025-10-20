import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps

@st.cache_data
def load_data():
    # Reads and parses the time in the first column directly
    return pd.read_csv("open-meteo-subset.csv", parse_dates=[0])

df = load_data()


st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page:", ["Home", "Data overview", "Plots", "Open space"])


if page == "Home":
    st.title("Welcome to the Weather Data App")
    st.write("This is a simple Streamlit-app which shows weather data from 2020, so for only for January.")
    st.write("Use the menu on the left to navigate between pages.")

    ASSETS = Path(__file__).parent / "assets"
    img_path = ASSETS / "img_8485.jpg"

    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # Rotate the image if necessary
    st.image(img, caption="Helvetestinden - Reine, by Maren Sofie Steen", use_container_width=True)


# --- Data overview ---
elif page == "Data overview":
    st.title("Data overview")
    st.write("Overview per variable with a mini-graph for the first month (0–744).")

    # Assume the first column is time
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # First month (~744 timer)
    df_month = df.iloc[:744].copy()

    # Build a table: one row per variable (skipping the time column)
    rows = []
    for col in df.columns[1:]:
        # Extract the unit in parentheses, ex. "(°C)"
        m = re.search(r"\((.*?)\)", col)
        unit = m.group(1) if m else ""

        var_name = re.sub(r"\s*\([^)]*\)\s*$", "", col).strip()  # remove everything from the parentheses and space between
        var_name_pretty = var_name.replace("_", " ").strip().capitalize()

        rows.append({
            "Variables": var_name_pretty,
            "Unit": unit,
            "First month": df_month[col].tolist(),  # <- sparkline
            "Min": float(df_month[col].min()),
            "Max": float(df_month[col].max()),
            "Mean": float(df_month[col].mean()),
        })

    spark_df = pd.DataFrame(rows)

    st.dataframe(
        spark_df,
        hide_index=True,
        column_config={
            "First month": st.column_config.LineChartColumn(
                label="Plot (first month)",
                width="large",
                y_min=None,
                y_max=None
            ),
            "Min": st.column_config.NumberColumn(format="%.2f"),
            "Max": st.column_config.NumberColumn(format="%.2f"),
            "Mean": st.column_config.NumberColumn(format="%.2f"),
        }
    )

    # Raw data as a preview
    with st.expander("Show raw data (first 744 rows)"):
        st.dataframe(df.head(744))


    
elif page == "Plots":
    st.title("Plots")

    # Assume the first column is time
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # --- Select column (selectbox: one column or All) ---
    data_cols = list(df.columns[1:])  # All numerical variables (assumed))
    col_choice = st.selectbox("Select variables:", ["All"] + data_cols)

    # --- Select month/interval (select_slider) ---
    # Assume that your dataset is for 2020.
    mnd_labels = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
    label2num = {lbl: i+1 for i, lbl in enumerate(mnd_labels)}

    # Default = First month (January → January). Use a tuple for interval support.
    m_start_lbl, m_end_lbl = st.select_slider(
        "Select month(s):",
        options=mnd_labels,
        value=("January", "January")  # default: First month
    )
    m_start, m_end = label2num[m_start_lbl], label2num[m_end_lbl]

    # Filter the selected month intervall (inclusive)
    mask = (df[time_col].dt.month >= m_start) & (df[time_col].dt.month <= m_end)
    df_sel = df.loc[mask].copy().set_index(time_col)

    # --- Plot ---
    if col_choice == "All":
        st.subheader(f"All variables – {m_start_lbl}–{m_end_lbl}")
        st.line_chart(df_sel[data_cols])  # all selected columns
    else:
        st.subheader(f"{col_choice} – {m_start_lbl}–{m_end_lbl}")
        st.line_chart(df_sel[[col_choice]])


elif page == "Open space":
    st.title("Empty page")
    st.write("""
    ## Not in use for now
    ...""")