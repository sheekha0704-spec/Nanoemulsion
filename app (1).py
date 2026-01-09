
import streamlit as st
import joblib
import pandas as pd
from fpdf import FPDF
import base64

# --- PDF GENERATION FUNCTION ---
def create_pdf(data, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Nano-Commercialize AI: Formulation Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Final Prediction: {result}", ln=True)
    pdf.ln(5)
    
    pdf.cell(200, 10, "Formulation Details:", ln=True)
    for key, value in data.items():
        pdf.cell(200, 10, f"- {key}: {value}", ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

st.title("🧪 Nano-Commercialize AI")

try:
    model = joblib.load('drug_nano_predictor_v1.pkl')
    features = joblib.load('model_features.pkl')
    st.sidebar.success("✅ Model Loaded")
    
    input_data = {}
    for col in features:
        input_data[col] = st.sidebar.number_input(f"{col}", value=0.0)

    if st.button("Generate Commercial Viability Report"):
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        status = "STABLE" if str(prediction) == "1" or "STABLE" in str(prediction).upper() else "UNSTABLE"
        
        st.success(f"Formulation Result: {status}")
        
        # PDF Download Button
        pdf_bytes = create_pdf(input_data, status)
        st.download_button(label="📥 Download PDF Report", 
                           data=pdf_bytes, 
                           file_name="Nano_Formulation_Report.pdf", 
                           mime="application/pdf")
except Exception as e:
    st.error(f"Error: {e}")
