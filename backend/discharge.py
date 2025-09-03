import ollama
from fpdf import FPDF
import streamlit as st

def generate_post_discharge_plan_safe(row):
    """
    Generate a post-discharge plan using Ollama AI model.
    Takes patient info from the row and formats a prompt for the LLM.
    Returns AI-generated text (safe with exception handling).
    """
    try:
        with st.spinner("Generating AI Post-Discharge Plan..."):
            # Prepare patient summary string
            patient_info = f"Name: {row.get('name','Unknown')}, Age: {row.get('agefactor',0)}, Risk Score: {row.get('risk_score',0):.2f}% ({row.get('risk_level','LOW')}), Readmission Probability: {row.get('readmit_prob',0)*100:.1f}% ({row.get('readmit_flag','Low Risk')})"
            
            # Create AI prompt
            prompt = f"Generate a clinician-friendly post-discharge plan for the patient:\n{patient_info}"
            
            # Call Ollama chat API with llama2 model
            response = ollama.chat(
                model="llama2:7b",
                messages=[
                    {"role": "system", "content": "You are a clinical assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content'].strip()
    except Exception as e:
        # Return error message if AI call fails
        return f"❌ Could not generate plan: {e}"


def create_pdf_from_text(patient_row, text, title="Post-Discharge Plan"):
    """
    Create a PDF file containing patient info and the AI-generated text.
    Returns PDF bytes (for download in Streamlit).
    """
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)

    # Patient info
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Patient Name: {patient_row.get('name','Unknown')}")
    pdf.multi_cell(0, 8, f"Age: {patient_row.get('agefactor',0)}")
    pdf.multi_cell(0, 8, f"Disease: {patient_row.get('disease','Unknown')}")
    pdf.ln(5)

    # Main AI-generated text
    pdf.multi_cell(0, 8, text)

    # Return PDF as bytes (latin1 encoding required by FPDF)
    return pdf.output(dest='S').encode('latin1')
