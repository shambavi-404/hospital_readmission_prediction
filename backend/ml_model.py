import pickle
import os
import streamlit as st

# Default path for the saved ML model
MODEL_PATH = "xgb_readmission_model.pkl"

def load_ml_model(model_path=MODEL_PATH):
    """
    Load a trained ML model from disk (pickle file).
    Displays success/warning messages in Streamlit sidebar.
    Returns the model object if found, otherwise None.
    """
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)   # Load model from file
        st.sidebar.success("✅ ML model loaded successfully.")
        return model
    else:
        st.sidebar.warning("⚠ ML model not found. Falling back to legacy calculations.")
        return None

def predict_readmission(ml_model, features):
    """
    Predict readmission probability using the ML model.
    Expects 'features' as input (numpy array / DataFrame).
    Returns probabilities if model supports predict_proba, else None.
    """
    try:
        return ml_model.predict_proba(features)[:,1]  # Probability of class=1
    except:
        return None   # Fallback if model doesn't support predict_proba
