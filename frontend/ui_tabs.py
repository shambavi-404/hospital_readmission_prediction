import streamlit as st
import plotly.graph_objects as go
import altair as alt

from report import generate_structured_report, create_patient_pdf_bytes
from discharge import generate_post_discharge_plan_safe, create_pdf_from_text
from shap_analysis import compute_shap_values
from root_cause import root_cause_explorer

def render_tabs(patient_row, patients_df, ml_model):
    """
    Render Streamlit tabs for patient analytics:
    1. Risk Assessment (gauge + recommendation)
    2. Penalty Reduction Simulation
    3. AI-Generated Summary
    4. SHAP Analysis
    5. Post-Discharge Plan
    6. Root-Cause Explorer
    """

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Risk Assessment",
        "🏥 Penalty Reduction",
        "📝 AI Summary",
        "🔬 SHAP Analysis",
        "📋 Post-Discharge Plan",
        "📊 Root-Cause Explorer"
    ])

    # -------------------------
    # Tab 1: Risk Assessment
    # -------------------------
    with tab1:
        st.markdown("## 🧾 Patient Risk Assessment")

        # Extract patient risk info
        risk_score = patient_row['risk_score']
        risk_level = patient_row['risk_level']
        recommendation = patient_row['recommendation']

        # Gauge chart for risk score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "tomato"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Show risk level + recommendation with styled UI
        risk_colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
        rec_colors = {
            "Extended monitoring": "#2980b9",
            "Immediate intervention": "#c0392b",
            "Routine check": "#27ae60"
        }

        st.markdown(
            f"""
            <h3 style="color:#d35400;">⚠️ Risk Level</h3>
            <span style="background:linear-gradient(135deg,{risk_colors.get(risk_level,'#34495e')},#ffffff20);
                        color:white;padding:8px 18px;border-radius:25px;font-size:20px;font-weight:bold;
                        box-shadow:0 2px 6px rgba(0,0,0,0.2);">{risk_level}</span>
            <h3 style="color:#16a085;margin-top:25px;">💡 Recommendation</h3>
            <span style="background:linear-gradient(135deg,#2980b9,#6dd5fa);
                        color:white;padding:8px 18px;border-radius:25px;font-size:18px;font-weight:bold;
                        box-shadow:0 2px 6px rgba(0,0,0,0.2);">{recommendation}</span>
            """,
            unsafe_allow_html=True
        )

    # -------------------------
    # Tab 2: Simulation / Penalty Reduction
    # -------------------------
    with tab2:
        st.markdown("## 🧪 Simulation Results")

        # Constants for cost/savings
        readmission_cost = 15000
        prevention_success_rate = 0.7
        high_risk_fraction = 0.2

        # Estimate savings per patient and hospital-wide
        patient_saving_sim = readmission_cost * patient_row['risk_score']/100 * prevention_success_rate
        total_patients = len(patients_df)
        high_risk_patients_sim = total_patients * high_risk_fraction
        hospital_saving_sim = high_risk_patients_sim * prevention_success_rate * readmission_cost
        max_penalty_sim = 26_000_000_000 * 0.15
        hospital_saving_sim = min(hospital_saving_sim, max_penalty_sim)

        # Display savings in 2 styled cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#27ae60,#2ecc71);
                            padding:20px;border-radius:15px;color:white;text-align:center;
                            box-shadow:0 4px 10px rgba(0,0,0,0.15);">
                    <h3 style="margin-bottom:5px;">👤 Individual Saving</h3>
                    <p style="font-size:26px;font-weight:bold;margin:0;">${patient_saving_sim:,.0f}</p>
                </div>""", unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#2980b9,#6dd5fa);
                            padding:20px;border-radius:15px;color:white;text-align:center;
                            box-shadow:0 4px 10px rgba(0,0,0,0.15);">
                    <h3 style="margin-bottom:5px;">🏥 Hospital-wide Potential</h3>
                    <p style="font-size:26px;font-weight:bold;margin:0;">${hospital_saving_sim:,.0f}</p>
                </div>""", unsafe_allow_html=True
            )

        # Notes about assumptions
        st.markdown(
            """
            <p style="font-size:14px;color:white;margin-top:15px;">
            💡 Estimates based on <b>prevention success rate (70%)</b> 
            and <b>average readmission cost ($15,000)</b>.  
            Hospital-wide savings capped at CMS maximum penalty.
            </p>""", unsafe_allow_html=True
        )

    # -------------------------
    # Tab 3: AI Summary
    # -------------------------
    with tab3:
        st.markdown("## 🤖 AI-Generated Patient Summary")

        # Generate structured report dictionary
        report = generate_structured_report(patient_row)

        # Colors/icons for report sections
        color_map = {"Clinical": "#e74c3c", "Behavioral": "#f39c12", "Social/System": "#2980b9", "Other": "#8e44ad"}
        icons = {"Clinical": "🩺", "Behavioral": "🧠", "Social/System": "🌍", "Other": "📌"}

        # Render report in styled cards
        for section, content in report.items():
            items_html = ""
            if isinstance(content, dict):
                for k, v in content.items():
                    items_html += f"<li><b>{k}:</b> {', '.join(v)}</li>"
            elif isinstance(content, list):
                for item in content:
                    items_html += f"<li>{item}</li>"
            else:
                items_html += f"<li>{content}</li>"

            st.markdown(
                f"""
                <div style="background:{color_map.get(section,'#34495e')};padding:15px;
                            border-radius:12px;margin-bottom:15px;color:white;
                            box-shadow:0 4px 10px rgba(0,0,0,0.2);">
                    <h4 style="margin:0;">{icons.get(section,'📄')} {section}</h4>
                    <ul style="margin:8px 0 0 15px;">{items_html}</ul>
                </div>
                """, unsafe_allow_html=True
            )

        # Export AI summary as PDF
        pdf_bytes = create_patient_pdf_bytes(patient_row, str(report))
        st.download_button(
            "⬇ Download Patient Summary PDF",
            data=pdf_bytes,
            file_name=f"{patient_row.get('name','unknown')}_summary.pdf",
            mime="application/pdf"
        )

    # -------------------------
    # Tab 4: SHAP Analysis
    # -------------------------
    with tab4:
        if ml_model:
            shap_values_df = compute_shap_values(ml_model, patients_df, patient_row['patient_id'])
            st.success("SHAP values computed. Run Root-Cause tab to explore.")
        else:
            st.info("SHAP analysis available only with ML model.")

    # -------------------------
    # Tab 5: Post-Discharge Plan
    # -------------------------
    with tab5:
        st.subheader("AI-Generated Post-Discharge Plan")

        # Generate plan via Ollama model
        discharge_plan = generate_post_discharge_plan_safe(patient_row)

        # Display + allow editing
        st.text_area("Plan", value=discharge_plan, height=400)

        # Export plan as PDF
        pdf_bytes = create_pdf_from_text(patient_row, discharge_plan, title="Post-Discharge Plan")
        st.download_button(
            "⬇ Download PDF",
            data=pdf_bytes,
            file_name=f"{patient_row.get('name','unknown')}_discharge_plan.pdf",
            mime="application/pdf"
        )

    # -------------------------
    # Tab 6: Root-Cause Explorer
    # -------------------------
    with tab6:
        # Show SHAP-based root cause explorer if available
        if ml_model and 'shap_values_df' in locals():
            root_cause_explorer(patient_row['patient_id'], shap_values_df)
        else:
            st.info("Run the SHAP Analysis tab first to generate SHAP values for this patient.")
