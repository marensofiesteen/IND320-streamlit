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
    bash: mkdir -p .streamlit

    [mongo]
    uri = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=mss2"
    db = "elhub"
    collection = "production_data"

5. **Start the app**
    streamlit run Home.py

**NB: Project structure**
    IND320-streamlit/
    ├─ .streamlit/secrets.toml   # not included in version control
    ├─ pages/                    # Streamlit subpages
    │   ├─ Production_Elhub.py
    │   ├─ STL_and_Spectrogram.py
    │   ├─ Data_overview.py
    │   ├─ Weather_plots.py
    │   └─ Outliers_and_LOF.py
    ├─ utils/
    │   ├─ utils_openmeteo.py    # Open-Meteo API client
    ├─ analysis_utils.py         # STL, spectrogram, SATV/SPC, LOF
    ├─ db_mongo.py               # MongoDB connection and queries
    ├─ requirements.txt
    ├─ Home.py                   # Global selector
    └─ README.md


**Prerequisites**
    Python 3.11+
    MongoDB Atlas cluster containt: elhub.production_data
    Streamlit 1.30+
    Internet connection (live Open-Meteo API requests)