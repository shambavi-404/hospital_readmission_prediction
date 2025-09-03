import streamlit as st

def render_patient_metrics(patient_row):
    """
    Render key patient metrics in the Streamlit dashboard:
    - Risk Score
    - Readmission Probability (flag)
    - Expected Saving
    Also shows a colored alert banner based on readmission risk.
    """

    # Display metrics in 3 columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk Score", f"{patient_row['risk_score']:.1f}%", patient_row['risk_level'])
    with col2:
        st.metric("Readmission Prob", patient_row['readmit_flag'])
    with col3:
        st.metric("Expected Saving", f"${patient_row['expected_saving']:,.0f}")

    # Choose alert color based on risk category
    alert_color = (
        "#e74c3c" if patient_row['readmit_flag']=="Will be readmitted" 
        else "#f39c12" if patient_row['readmit_flag']=="May be readmitted" 
        else "#27ae60"
    )

    # Render styled HTML alert box
    st.markdown(
        f"""
        <div style="padding:12px; border-radius:10px; background-color:{alert_color};
                    color:white; font-weight:bold; text-align:center;">
            ⚠ Predictive Readmission Alert: {patient_row['readmit_flag']} 
            ({patient_row['readmit_prob']*100:.1f}%)
        </div>
        """,
        unsafe_allow_html=True
    )
