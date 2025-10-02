import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


file_path = "open-meteo-subset.csv"
df = pd.read_csv(file_path)


st.sidebar.title("Navigasjon")
page = st.sidebar.radio("Velg side:", ["Hjem", "Dataoversikt", "Plott", "open space"])


if page == "Hjem":
    st.title("Velkommen til værdata-appen")
    st.write("Dette er en enkel Streamlit-app som viser værdata for 2020, foreløpig kun for januar.")
    st.write("Bruk menyen til venstre for å navigere mellom sidene.")


# --- Dataoversikt ---
elif page == "Dataoversikt":
    st.title("Dataoversikt")
    st.write("Her er en oversikt over værdataene for januar:")

    # Ta første 745 rader (~ en måned)
    df_month = df.iloc[:744]

    st.dataframe(df_month)


elif page == "Plott":
    st.title("Plott")

    # Ta første 745 rader (~ en måned)
    df_month = df.iloc[:744]


    st.write("Mini-plots per kolonne (første måned):")

# Loop gjennom alle variabler unntatt første kolonne (tid)
    for col in df.columns[1:]:
        st.subheader(f"{col} Januar")

        fig, ax = plt.subplots()
        ax.plot(df_month.index, df_month[col])  # eller bruk df_month[tid] hvis du har en tid-kolonne
        ax.set_xlabel("Observasjon")
        ax.set_ylabel(col)
        st.pyplot(fig)
    columns_options = ["Alle"] + list(df.columns[1:])
    selected_column = st.selectbox("Velg kolonne:", columns_options)


    min_row = int(df.index.min())
    max_row = int(df.index.max())
    selected_range = st.select_slider(
        "Velg radområde:",
        options=list(range(min_row, max_row + 1)),
        value=(min_row, min_row + 744)  # Default to first month
    )


    df_subset = df.iloc[selected_range[0]:selected_range[1] + 1]


    if selected_column == "Alle":
        st.line_chart(df_subset.set_index(df_subset.columns[0]))
    else:
        st.line_chart(df_subset[[selected_column]].set_index(df_subset.columns[0]))


elif page == "open space":
    st.title("Tomt")
    st.write("""
    ## Foreløpig ikke i bruk
    ...""")