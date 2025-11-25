# 📊 IND320 – Energy, Meteorology & Production Dashboard

A Streamlit-based analytics dashboard for exploring, visualizing and forecasting  
Norwegian Elhub energy production/consumption data together with Open-Meteo ERA5 weather data.

This app is structured into four logical groups:

1. **Explorative / Overview** – inspect energy and weather data  
2. **Anomalies / Data Quality** – SPC, LOF, and unusual patterns  
3. **Forecasting** – SARIMAX forecasting with optional exogenous weather variables  
4. **Snow & Geo** – snow drift analysis, wind roses, and map visualizations  


## 📦 Installation

1. **Clone the project**
   ```bash
   git clone https://github.com/<username>/IND320-streamlit.git
   cd IND320-streamlit

2. **Create and activate a virtual environment**
    python -m venv .venv
    source .venv/bin/activate   # macOS/Linux
    .venv\Scripts\activate      # Windows

3. **Install the required packages**
    pip install -r requirements.txt

4. **Create the file .streamlit/secrets.toml**
    Make sure the folder exists:
    ```bash
    mkdir -p .streamlit

    Create the file .streamlit/secrets.toml:
    [mongo]
    uri = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=mss2"
    db = "elhub"
    collection = "production_data"

5. **Start the app**
    streamlit run Home.py

**NB: Project structure**
    IND320-streamlit/
    ├─ .streamlit/
    │   ├─ secrets.toml   # not included in version control
    ├─ pages/                    # Streamlit subpages
    │   ├─ 2_Production_Elhub.py
    │   ├─ 3_STL_and_Spectrogram.py
    │   ├─ 4_Data_overview.py
    │   ├─ 5_Weather_Plots.py
    │   ├─ 6_Outliers_and_LOF.py
    │   ├─ 8_Map_price_areas.py
    │   ├─ 9_Snow_drift.py
    │   ├─ 10_Meteo_Energy_Correlation.py
    │   └─ 11_Forecasting_SARIMAX.py
    ├─ utils/
    │   ├─ utils_openmeteo.py    # Open-Meteo API client
    │   └─ navigation.py
    ├─ analysis_utils.py         # STL, spectrogram, SATV/SPC, LOF
    ├─ db_mongo.py               # MongoDB connection and queries
    ├─ elhub_mongo_utils.py
    ├─ Home.py                   # Global selector
    ├─ README.md
    └─ requirements.txt


**Prerequisites**
    Python 3.11+
    MongoDB Atlas cluster containt: 
        elhub.production_data and elhub.sonsumption_data
    Streamlit 1.30+
    Internet connection (live Open-Meteo API requests)