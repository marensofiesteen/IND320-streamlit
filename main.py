import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps

@st.cache_data
def load_data():
    # Leser og parser tid i første kolonne direkte
    return pd.read_csv("open-meteo-subset.csv", parse_dates=[0])

df = load_data()


st.sidebar.title("Navigasjon")
page = st.sidebar.radio("Velg side:", ["Hjem", "Dataoversikt", "Plott", "Open space"])


if page == "Hjem":
    st.title("Velkommen til værdata-appen")
    st.write("Dette er en enkel Streamlit-app som viser værdata for 2020, foreløpig kun for januar.")
    st.write("Bruk menyen til venstre for å navigere mellom sidene.")

    ASSETS = Path(__file__).parent / "assets"
    img_path = ASSETS / "img_8485.jpg"

    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # roter bilde hvis nødvendig
    st.image(img, caption="Helvetestinden - Reine, av Maren Sofie Steen", use_container_width=True)


# --- Dataoversikt ---
elif page == "Dataoversikt":
    st.title("Dataoversikt")
    st.write("Oversikt per variabel med minigraf for første måned (0–744).")

    # Anta første kolonne er tid
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Første måned (~744 timer)
    df_month = df.iloc[:744].copy()

    # Bygg tabell: én rad per variabel (hopper over tidskolonnen)
    rows = []
    for col in df.columns[1:]:
        # hent enhet i parentes, f.eks. "(°C)"
        m = re.search(r"\((.*?)\)", col)
        unit = m.group(1) if m else ""

        var_name = re.sub(r"\s*\([^)]*\)\s*$", "", col).strip()  # fjern alt i parentes + evt. mellomrom
        var_name_pretty = var_name.replace("_", " ").strip().capitalize()

        rows.append({
            "Variabel": var_name_pretty,
            "Enhet": unit,
            "Første måned": df_month[col].tolist(),  # <- sparkline
            "Min": float(df_month[col].min()),
            "Maks": float(df_month[col].max()),
            "Snitt": float(df_month[col].mean()),
        })

    spark_df = pd.DataFrame(rows)

    st.dataframe(
        spark_df,
        hide_index=True,
        column_config={
            "Første måned": st.column_config.LineChartColumn(
                label="Plott (første måned)",
                width="large",
                y_min=None,  # sett tall hvis du vil låse aksen
                y_max=None
            ),
            "Min": st.column_config.NumberColumn(format="%.2f"),
            "Maks": st.column_config.NumberColumn(format="%.2f"),
            "Snitt": st.column_config.NumberColumn(format="%.2f"),
        }
    )

    # (Valgfritt) rå-data som forhåndsvisning
    with st.expander("Vis rå importert data (første 744 rader)"):
        st.dataframe(df.head(744))


    
elif page == "Plott":
    st.title("Plott")

    # Forutsetter at første kolonne er tid
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # --- Velg kolonne (selectbox: én kolonne eller Alle) ---
    data_cols = list(df.columns[1:])  # alle numeriske variabler (antatt)
    col_choice = st.selectbox("Velg variabel:", ["Alle"] + data_cols)

    # --- Velg måned/intervall (select_slider) ---
    # Antar at datasettet ditt er for 2020. Juster hvis flere år.
    mnd_labels = ["Januar", "Februar", "Mars", "April", "Mai", "Juni",
                  "Juli", "August", "September", "Oktober", "November", "Desember"]
    label2num = {lbl: i+1 for i, lbl in enumerate(mnd_labels)}

    # Default = første måned (Januar → Januar). Bruk tuple for intervall-støtte.
    m_start_lbl, m_end_lbl = st.select_slider(
        "Velg måned(er):",
        options=mnd_labels,
        value=("Januar", "Januar")  # default: første måned
    )
    m_start, m_end = label2num[m_start_lbl], label2num[m_end_lbl]

    # Filtrer valgt månedsintervall (inklusive)
    mask = (df[time_col].dt.month >= m_start) & (df[time_col].dt.month <= m_end)
    df_sel = df.loc[mask].copy().set_index(time_col)

    # --- Plot ---
    if col_choice == "Alle":
        st.subheader(f"Alle variabler – {m_start_lbl}–{m_end_lbl}")
        st.line_chart(df_sel[data_cols])  # alle valgte kolonner
    else:
        st.subheader(f"{col_choice} – {m_start_lbl}–{m_end_lbl}")
        st.line_chart(df_sel[[col_choice]])


elif page == "Open space":
    st.title("Tomt")
    st.write("""
    ## Foreløpig ikke i bruk
    ...""")