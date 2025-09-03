import numpy as np

def calculate_risk_score(df):
    """
    Calculate patient risk scores using weighted factors:
    - Age
    - White blood cell count
    - Heart rate
    - Presence of diabetes / hypertension
    Applies a logistic transformation to normalize values into 0–100 range.
    Returns updated DataFrame with 'risk_score' column.
    """
    df['risk_score'] = (
        0.4 * (df['agefactor'] / 100) +       # Age contribution
        0.8 * (df['WBC mean'] / 20000) +      # WBC contribution
        0.3 * (df['heart rate'] / 200) +      # HR contribution
        1.5 * df['diabetes'] +                # Diabetes (binary)
        1.0 * df['hypertension']              # Hypertension (binary)
    )
    # Apply logistic function → scale into percentage
    df['risk_score'] = 1 / (1 + np.exp(-df['risk_score'])) * 100
    return df


def assign_risk_level(risk_score):
    """
    Assign categorical risk level based on score.
    - HIGH: ≥ 75
    - MEDIUM: ≥ 50
    - LOW: otherwise
    """
    if risk_score >= 75: 
        return "HIGH"
    elif risk_score >= 50: 
        return "MEDIUM"
    else: 
        return "LOW"


def individual_savings(risk_score, cost_per_patient=15000, prevention_success_rate=0.7):
    """
    Estimate cost savings per patient if readmission is prevented.
    Formula: cost × risk probability × prevention success rate.
    """
    readmit_prob = risk_score / 100
    return cost_per_patient * readmit_prob * prevention_success_rate


def hospital_impact(df):
    """
    Calculate expected hospital-wide savings:
    - Compute savings for each patient
    - Sum across all patients
    - Cap at CMS maximum penalty reduction (15% of $26B)
    Returns total capped savings.
    """
    df['expected_saving'] = df['risk_score'].apply(individual_savings)
    total_saving = df['expected_saving'].sum()
    max_penalty_reduction = 26e9 * 0.15
    return min(total_saving, max_penalty_reduction)
