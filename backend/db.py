import sqlite3
import pandas as pd
import streamlit as st

# Path to the local SQLite database file
DB_PATH = "ehr_large.db"

@st.cache_data
def load_patients_from_db(db_path=DB_PATH):
    """
    Load patient data from the SQLite database.
    Uses Streamlit's cache to avoid reloading on every run.
    """
    conn = sqlite3.connect(db_path)   # Connect to SQLite DB
    try:
        # Read admissions_scored table into a pandas DataFrame
        df = pd.read_sql("SELECT * FROM admissions_scored", conn)
    finally:
        conn.close()   # Ensure DB connection is closed
    return df   # Return patient records as DataFrame
