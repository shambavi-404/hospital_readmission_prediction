import streamlit as st
import os
import pandas as pd
import pickle

# ✅ Corrected imports from local modules
from db import load_patients_from_db
from ml_model import load_ml_model
from frontend import ui_tabs

# -------------------------
# Page Config
# -------------------------
st.set_page_config(layout="wide")
st.title("🏥 Hospital EHR Patient Analysis")

# -------------------------
# Sidebar: Hospital Parameters
# -------------------------
st.sidebar.header("Hospital Parameters")
DB_PATH = "ehr_large.db"
MODEL_PATH = "xgb_readmission_model.pkl"

# Load patient dataset + ML model
patients_df = load_patients_from_db(DB_PATH)
ml_model = load_ml_model(MODEL_PATH)

# Sidebar inputs for hospital-level parameters
total_patients = st.sidebar.number_input(
    "Total patients monthly", 
    value=max(1000, len(patients_df)), 
    step=1
)
current_read_rate = st.sidebar.number_input(
    "Current readmission rate (0-1)", 
    value=0.15, 
    step=0.01
)

# -------------------------
# Patient Selection
# -------------------------
# Build patient lookup (IDs + names)
patient_ids = patients_df['patient_id'].tolist()
patient_names = patients_df['name'].astype(str).tolist() if 'name' in patients_df.columns else [str(x) for x in patient_ids]
id_to_name = dict(zip(patient_ids, patient_names))

# Prepare search-friendly DataFrame
df_search = patients_df.copy()
df_search['patient_id_str'] = df_search['patient_id'].astype(str)
df_search['name_str'] = df_search['name'].astype(str) if 'name' in df_search.columns else df_search['patient_id_str']
df_search['patient_id_str_l'] = df_search['patient_id_str'].str.lower()
df_search['name_str_l'] = df_search['name_str'].str.lower()
df_search['prefixed_id_l'] = ('patient_' + df_search['patient_id_str']).str.lower()

# Search box for patient lookup
q = st.text_input("🔍 Search Patient (ID or Name)", placeholder="e.g., 770, Patient_770, or patient name").strip()
q_l = q.lower()
selected_patient_id = None

# Handle exact and partial matches
if q:
    exact_mask = (
        (df_search['patient_id_str_l'] == q_l) |
        (df_search['prefixed_id_l'] == q_l) |
        (df_search['name_str_l'] == q_l)
    )
    exact_hits = df_search.loc[exact_mask]

    if len(exact_hits) == 1:
        # ✅ Unique exact match
        selected_patient_id = exact_hits['patient_id'].iloc[0]
        st.success(f"✅ Found exact match: {id_to_name.get(selected_patient_id, selected_patient_id)} (ID: {selected_patient_id})")
    elif len(exact_hits) > 1:
        # ⚠ Multiple exact matches
        options = exact_hits['patient_id'].tolist()
        selected_patient_id = st.selectbox("Select matching patient", options, format_func=lambda pid: f"{id_to_name.get(pid, str(pid))} (ID: {pid})")
    else:
        # 🔍 Fallback to partial search
        partial_mask = (
            df_search['patient_id_str_l'].str.contains(q_l, na=False) |
            df_search['prefixed_id_l'].str.contains(q_l, na=False) |
            df_search['name_str_l'].str.contains(q_l, na=False)
        )
        partial_hits = df_search.loc[partial_mask]
        if len(partial_hits) >= 1:
            options = partial_hits['patient_id'].tolist()
            selected_patient_id = st.selectbox("Select matching patient", options, format_func=lambda pid: f"{id_to_name.get(pid, str(pid))} (ID: {pid})")
        else:
            st.warning("⚠ No matching patient found. Showing full list below.")
            selected_patient_id = st.selectbox("Select patient", patient_ids, format_func=lambda pid: id_to_name.get(pid, str(pid)))
else:
    # Default: show full patient list
    selected_patient_id = st.selectbox("Select patient", patient_ids, format_func=lambda pid: id_to_name.get(pid, str(pid)))

# Fetch selected patient row
patient_row = patients_df.loc[patients_df['patient_id'] == selected_patient_id].iloc[0]

# -------------------------
# Render Tabs (frontend module)
# -------------------------
ui_tabs.render_tabs(patient_row, patients_df, ml_model)
