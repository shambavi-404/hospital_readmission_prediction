import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.graph_objects as go

# -------------------------
# Mapping of features → categories
# -------------------------
feature_categories = {
    'agefactor': 'Clinical',
    'WBC mean': 'Clinical',
    'heart rate': 'Clinical',
    'diabetes': 'Clinical',
    'hypertension': 'Clinical',
    'missed_followups': 'Behavioral',
    'med_adherence': 'Behavioral',
    'lives_alone': 'Social/System',
    'distance_to_hospital': 'Social/System',
    'insurance_status': 'Social/System'
}

def root_cause_explorer(patient_id, shap_values_df):
    """
    Root Cause Explorer for SHAP values.
    - Shows feature-level impact (waterfall chart)
    - Summarizes category-level impact (pie chart)
    - Provides doctor-friendly explanations
    - Allows simulation of modifiable risk factors
    """

    # -------------------------
    # Fetch SHAP values for given patient
    # -------------------------
    patient_shap = shap_values_df[shap_values_df['patient_id'] == patient_id].copy()
    if patient_shap.empty:
        st.warning("No SHAP values found for this patient.")
        return

    # Add feature → category mapping
    patient_shap['Category'] = patient_shap['feature'].map(feature_categories).fillna('Other')

    # Summarize contributions by category
    category_summary = (
        patient_shap.groupby('Category')['shap_value']
        .apply(lambda x: np.sum(np.abs(x)))
        .reset_index()
        .rename(columns={'shap_value': 'Total_Impact'})
    )

    # ==================================================
    # 🔝 Top Drivers - Waterfall Chart
    # ==================================================
    st.markdown("### 🔝 Top Drivers of Readmission Risk (Waterfall View)")

    # Select top 5 most influential features
    top_features = (
        patient_shap.sort_values(by="shap_value", key=lambda x: abs(x), ascending=False)
        .head(5)
    )

    # Build Plotly Waterfall
    fig = go.Figure(
        go.Waterfall(
            name="Risk Impact",
            orientation="v",
            measure=["relative"] * len(top_features),
            x=top_features["feature"],
            text=[f"{val:.2f}" for val in top_features["shap_value"]],
            y=top_features["shap_value"],
            connector={"line": {"color": "rgba(63, 63, 63, 0.5)"}},
            textposition="outside",
            increasing={"marker": {"color": "salmon"}},     # ↑ Risk
            decreasing={"marker": {"color": "lightgreen"}}, # ↓ Risk
        )
    )

    fig.update_layout(
        title="Feature Contributions to Patient’s Readmission Risk",
        xaxis_title="Feature",
        yaxis_title="Impact on Risk Score",
        showlegend=False,
        waterfallgap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📝 Explanation for clinicians
    st.info("""
    *How to read this chart:*
    - Each bar shows how much a feature pushes the patient’s readmission risk *up (red)* or *down (green)*.
    - Longer bars = stronger influence.
    - Example: If missed_followups and HbA1c have tall red bars, they strongly ↑ risk.
    - A green bar (e.g., med_adherence) means it’s protective.
    """)

    # ==================================================
    # 🗂 Category Contribution - Pie Chart
    # ==================================================
    st.markdown("### 🗂 Category-wise Risk Contribution")

    pie_chart = (
        alt.Chart(category_summary)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta('Total_Impact:Q'),
            color=alt.Color('Category:N', scale=alt.Scale(scheme='set1')),
            tooltip=['Category', 'Total_Impact']
        )
    )
    st.altair_chart(pie_chart, use_container_width=True)

    # 📝 Explanation for clinicians
    st.info("""
    *How to read this chart:*
    - Each slice = how much a risk category contributes (Clinical, Behavioral, Social/System).
    - Bigger slice = stronger impact.
    - Helps prioritize interventions (e.g., focus on behavioral if that slice dominates).
    """)

    # Highlight the biggest driver
    top_category = category_summary.sort_values("Total_Impact", ascending=False).iloc[0]
    st.warning(f"⚠ Main driver: *{top_category['Category']}* "
               f"({top_category['Total_Impact']:.2f} impact on risk)")

    # Special note if "Other" is top driver
    if top_category['Category'] == "Other":
        st.info("""
        *Note for Doctors:*
        This patient’s main driver falls under *'Other'*,
        meaning the model identified less common lab/medical factors not in standard categories.
        """)

    # ==================================================
    # 🔄 Simulation of Modifiable Factors
    # ==================================================
    st.markdown("### 🔄 Simulate Modifiable Factors")

    modifiable_features = ['diabetes', 'hypertension', 'med_adherence', 'missed_followups']
    simulated_shap = top_features.copy()

    for feat in modifiable_features:
        if feat in top_features['feature'].values:
            current_val = st.slider(f"{feat} level", 0, 1, 1)
            simulated_shap.loc[simulated_shap['feature'] == feat, 'shap_value'] *= current_val

    simulated_risk_change = np.sum(np.abs(simulated_shap['shap_value']))
    st.metric("Simulated Total Risk Contribution", f"{simulated_risk_change:.2f}")

    # 📝 Explanation for clinicians
    st.info(f"""
    *Interpretation:*
    This value ({simulated_risk_change:.2f}) shows how much modifiable factors (diabetes, BP, adherence, follow-ups)
    contribute to the patient’s risk.
    - Adjust sliders to simulate improvements (e.g., better adherence ↓ risk contribution).
    """)



    # Call the explorer for the selected patient
    if ml_model and 'shap_values_df' in globals():
        root_cause_explorer(selected_patient_id, shap_values_df)
    else:
        st.info("Run the SHAP Analysis tab first to generate SHAP values for this patient.")
