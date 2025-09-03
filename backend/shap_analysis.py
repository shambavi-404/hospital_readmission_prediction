import shap
import pandas as pd
import numpy as np

def compute_shap_values(ml_model, patients_df, patient_id):
    """
    Compute SHAP values for a single patient.
    - Ensures required model features exist in the patient dataset
    - Uses SHAP TreeExplainer (for tree-based models like XGBoost)
    - Returns DataFrame with feature contributions to risk score

    Args:
        ml_model: Trained ML model (e.g., XGBoostClassifier)
        patients_df: DataFrame containing patient records
        patient_id: ID of patient to analyze

    Returns:
        shap_values_df: DataFrame with columns:
            - feature (str): feature name
            - shap_value (float): SHAP contribution for this patient
            - patient_id (int/str): patient identifier
    """

    # Extract the feature names expected by the ML model
    feature_cols = ml_model.get_booster().feature_names

    # Ensure all required features exist in patients_df (fill missing as 0)
    for col in feature_cols:
        if col not in patients_df.columns:
            patients_df[col] = 0

    # Select only the target patient's features
    X_patient = patients_df.loc[
        patients_df['patient_id'] == patient_id,
        feature_cols
    ]

    # Build SHAP explainer for the ML model
    explainer = shap.TreeExplainer(ml_model)

    # Compute SHAP values for this patient
    shap_values = explainer.shap_values(X_patient)

    # Wrap results into a DataFrame
    shap_values_df = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': shap_values[0],   # SHAP values for patient
        'patient_id': patient_id
    })

    return shap_values_df
