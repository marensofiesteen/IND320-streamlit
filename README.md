## 📦 Installasjon

1. **Clone the project**
   ```bash
   git clone https://github.com/<brukernavn>/IND320-streamlit.git
   cd IND320-streamlit

2. **Create and activate a virtual environment**
    python -m venv .venv
    source .venv/bin/activate   # macOS/Linux
    # .venv\Scripts\activate    # Windows

3. **Install the required packages**
    pip install -r requirements.txt

4. **Create the file .streamlit/secrets.toml**
    [mongo]
    uri = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=mss2"
    db = "elhub"
    collection = "production_data"

5. **Start the app**
    streamlit run main.py

**NB: Project structure**
    IND320-streamlit/
    ├─ main.py                # Hovedfil for Streamlit
    ├─ pages/                 # Undersider
    ├─ db_mongo.py            # Tilkobling til MongoDB Atlas
    ├─ utils/                 # Hjelpefunksjoner
    ├─ .streamlit/secrets.toml
    ├─ .gitignore
    └─ README.md

**Prerequisites**
    Python 3.11+
    MongoDB Atlas med elhub.production_data
    Streamlit 1.30+ installert