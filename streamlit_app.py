import streamlit as st
import pandas as pd

file_path = "/Users/marenssteen/Documents/IND320/Streamlit/open-meteo-subset.csv"
df = pd.read_csv(file_path)

st.title("Min app")
st.write("Her er dataene mine:")
st.dataframe(df.head())