import streamlit as st

st.set_page_config(page_title="Home", layout="wide")
st.title("Home")

st.markdown(""" 
## Welcome to the Weather Data App!

This app allows you to explore data. You can view an overview of the data, create plots for different variables, and analyze electricity production data from Elhub.

### Navigation
Use the sidebar on the left to navigate between different pages:
- **Home**: Introduction to the app.
- **Area & Year Selector**: Choose a price area and year for analysis.
- **STL & Spectrogram**: Analyze seasonality and time-frequency content of Elhub production data.
- **Data overview**: View a summary of the weather data with mini-graphs.
- **Plots**: Create line plots for selected weather variables over specified months.
- **Outliers and LOF**: Detect and visualize outliers in the weather data.
- **Production Elhub**: Analyze electricity production data from Elhub
""")