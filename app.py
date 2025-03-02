import streamlit as st
import datetime
from fpdf import FPDF
from transformers import pipeline

# Load AI model (Optimized)
try:
    legal_ai = pipeline("text-generation", model="gpt2-medium")
except:
    legal_ai = pipeline("text-generation", model="gpt2")  # Fallback to CPU

# Function to generate structured legal text
def generate_legal_text(facts, arguments, demand, legal_consequence):
    formatted_args = "\n".join([f"{i+1}. {arg}" for i, arg in enumerate(arguments) if arg])
    
    prompt = f"""
    Legal Demand Letter:
    - Background: {facts}
    - Legal Basis: {formatted_args}
    - Demand: {demand}
    - Legal Consequences: {legal_consequence}
    Write this in a professional, structured format.
    """
    
    try:
        response = legal_ai(prompt, max_length=200, num_return_sequences=1, truncation=True)
        return response[0]["generated_text"]
    except Exception as e:
        return f"Error generating legal text: {e}"

# Function to generate PDF
def generate_demand_letter(ref_our, ref_your, defendant, subject, client, facts, arguments, demand, legal_consequence, deadline, advocate):
    legal_text = generate_legal_text(facts, arguments, demand, legal_consequence)
    today_date = datetime.datetime.now().strftime("%d %B %Y")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.cell(200, 10, txt="DEMAND LETTER", ln=True, align='C')
    pdf.ln(10)
    
    # References & Date
    pdf.cell(200, 10, txt=f"Our Ref: {ref_our}     Your Ref: {ref_your}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Date: {today_date}", ln=True)
    pdf.ln(10)

    # Defendant Address
    pdf.multi_cell(0, 10, txt=defendant)
    pdf.ln(5)

    # Subject
    pdf.cell(200, 10, txt=f"Subject: {subject}", ln=True)
    pdf.ln(10)

    # Introduction
    pdf.multi_cell(0, 10, txt=f"Dear {defendant},\n\nWe represent {client} regarding {subject}. Below are the details of this claim:")
    pdf.ln(5)

    # Facts
    pdf.multi_cell(0, 10, txt=f"**Background:**\n{facts}")
    pdf.ln(5)

    # Legal Basis
    if arguments:
        pdf.multi_cell(0, 10, txt="**Legal Basis:**")
        for i, arg in enumerate(arguments):
            if arg:
                pdf.multi_cell(0, 10, txt=f"{i+1}. {arg}")
        pdf.ln(5)

    # AI-generated legal text
    pdf.multi_cell(0, 10, txt="**Legal Interpretation:**")
    pdf.multi_cell(0, 10, txt=legal_text)
    pdf.ln(5)

    # Demand
    pdf.multi_cell(0, 10, txt=f"**Demand:**\n{demand}")
    pdf.ln(5)

    # Legal Consequences
    pdf.multi_cell(0, 10, txt=f"**Legal Consequences:**\n{legal_consequence}")
    pdf.ln(5)

    # Deadline
    pdf.multi_cell(0, 10, txt=f"Compliance with this demand is expected within {deadline} days. Failure to do so will result in immediate legal action.")
    pdf.ln(10)

    # Advocate Signature
    pdf.multi_cell(0, 10, txt=f"**Sincerely,**\n\n{advocate}")

    # Save PDF
    pdf_path = "demand_letter.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Streamlit UI
st.title("📜 AI-Powered Demand Letter Generator")

ref_our = st.text_input("Our Reference")
ref_your = st.text_input("Your Reference")
defendant = st.text_area("Defendant's Name & Address")
subject = st.text_input("Subject Matter")
client = st.text_input("Client Name")
facts = st.text_area("Facts of the Case")

# Flexible Argument Inputs
arguments = st.text_area("Enter Arguments (One per Line)").split("\n")

demand = st.text_area("Demand/Remedy")
legal_consequence = st.text_area("Legal Consequences")
deadline = st.number_input("Time for Compliance (Days)", min_value=1, value=7)
advocate = st.text_input("Advocate's Name & Signature")

if st.button("Generate Demand Letter"):
    pdf_path = generate_demand_letter(ref_our, ref_your, defendant, subject, client, facts, arguments, demand, legal_consequence, deadline, advocate)
    st.success("✅ PDF Generated Successfully!")
    with open(pdf_path, "rb") as pdf_file:
        st.download_button("Download Demand Letter", pdf_file, file_name="demand_letter.pdf", mime="application/pdf")
