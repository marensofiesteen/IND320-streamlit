import pandas as pd
from pymongo import MongoClient
import streamlit as st

@st.cache_data(show_spinner=True)
def load_mongo_collection(collection_name: str) -> pd.DataFrame:
    """
    Connect to MongoDB using Streamlit secrets and load all documents
    from the given collection into a pandas DataFrame.
    The _id field is removed to keep the DataFrame clean.
    """
    mongo_secrets = st.secrets["mongo"]
    uri = mongo_secrets["uri"]
    db_name = mongo_secrets["db"]

    client = MongoClient(uri)
    db = client[db_name]
    coll = db[collection_name]

    docs = list(coll.find({}, {"_id": 0}))
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)

    # Try to parse datetime columns if present
    for col in ["starttime", "endtime", "lastupdatedtime", "startTime", "endTime", "lastUpdatedTime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df
