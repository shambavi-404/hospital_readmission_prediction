from fpdf import FPDF

def generate_structured_report(row):
    """
    Generate a structured dictionary report for a patient.
    Includes:
    - Patient summary
    - Risk factors
    - Medications & suggestions
    - Recommended interventions
    - Clinical notes
    """
    report = {}

    # -------------------------
    # Patient summary
    # -------------------------
    readmit_prob_pct = row.get('readmit_prob', 0) * 100
    readmit_flag = row.get('readmit_flag', 'Low Risk')
    report['Patient Summary'] = (
        f"{row.get('name','Unknown')}, Age {row.get('agefactor',0)}, "
        f"Risk Score: {row.get('risk_score',0):.2f}% ({row.get('risk_level','LOW')}), "
        f"ML Readmission Probability: {readmit_prob_pct:.1f}% ({readmit_flag})"
    )

    # -------------------------
    # Risk factors
    # -------------------------
    factors = []
    if row.get('agefactor',0) > 65:
        factors.append("Advanced age")
    for c in ['diabetes','hypertension','ckd','copd','cad','stroke','cancer']:
        if row.get(c,0) == 1:
            factors.append(c.capitalize())
    if row.get('WBC mean',0) > 11000:
        factors.append(f"High WBC ({row.get('WBC mean',0):,.0f})")
    if row.get('heart rate',0) > 100:
        factors.append(f"High HR ({row.get('heart rate',0):.0f} bpm)")
    if row.get('BP-mean',0) > 140 or row.get('BP-mean',0) < 90:
        factors.append(f"Abnormal BP ({row.get('BP-mean',0):.0f} mmHg)")
    factors.append(f"Readmission Risk: {readmit_flag}")
    report['Risk Factors'] = factors if factors else ["None notable"]

    # -------------------------
    # Medications & suggestions
    # -------------------------
    meds = []
    suggestions = []
    for m, suggestion in [
        ('antibiotics', "Consider antibiotics if infection suspected"),
        ('antihypertensives', "Ensure BP control with antihypertensives"),
        ('insulin', "Manage glucose with insulin or oral agents"),
        ('statins', "Continue statins for cardiac risk"),
        ('anticoagulants', "Evaluate need for anticoagulation")
    ]:
        if row.get(m,0):
            meds.append(m.capitalize())
        elif m == 'antibiotics' and row.get('WBC mean',0) > 11000:
            suggestions.append("Consider antibiotics for elevated WBC")

    report['Medications & Suggestions'] = {
        'Current Medications': meds if meds else ["None"], 
        'Suggestions': suggestions if suggestions else ["No additional suggestions"]
    }

    # -------------------------
    # Recommended interventions
    # -------------------------
    report['Recommended Interventions'] = row.get('recommendation','Standard follow-up')

    # -------------------------
    # Clinical notes
    # -------------------------
    notes = []
    if row.get('temperature mean',0) > 100.4:
        notes.append("Monitor for fever")
    if row.get('haemoglobin',0) < 12:
        notes.append("Check for anemia")
    report['Notes for Clinicians'] = notes if notes else ["No immediate concerns"]

    return report


def create_patient_pdf_bytes(patient_row, summary_text):
    """
    Generate a PDF summary for a patient.
    Includes demographics, risk score, recommendation, and AI summary.
    Returns PDF as bytes (latin1 encoding required by FPDF).
    """
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Patient Risk Summary", ln=True, align="C")

    # Patient info
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.multi_cell(0, 8, f"Patient Name: {patient_row.get('name','Unknown')}")
    pdf.multi_cell(0, 8, f"Age: {patient_row.get('agefactor',0)}")
    pdf.multi_cell(0, 8, f"Disease: {patient_row.get('disease','Unknown')}")
    pdf.multi_cell(0, 8, f"Risk Score: {patient_row.get('risk_score',0):.2f}% ({patient_row.get('risk_level','LOW')})")
    pdf.multi_cell(0, 8, f"Recommendation: {patient_row.get('recommendation','Standard follow-up')}")

    # AI-generated summary
    pdf.ln(5)
    pdf.multi_cell(0, 8, "AI-Generated Summary:")
    pdf.multi_cell(0, 8, summary_text)

    return pdf.output(dest='S').encode('latin1')
