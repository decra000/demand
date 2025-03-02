import torch

# Only clear CUDA cache if a GPU is available
if torch.cuda.is_available():
    torch.cuda.empty_cache()


import gradio as gr  

# Import FPDF for PDF generation
from fpdf import FPDF  

# Import Transformers for AI models
from transformers import pipeline  

# Import Torch & related libraries for deep learning support
import torchvision  
import torchaudio  

# Clear GPU cache (if using GPU)
torch.cuda.empty_cache()  

import datetime

# Clear GPU cache
torch.cuda.empty_cache()

# Load AI model (Optimized)
try:
    legal_ai = pipeline("text-generation", model="gpt2-medium", device=0)  # Use GPU if available
except:
    legal_ai = pipeline("text-generation", model="gpt2")  # Fallback to CPU

# Function to generate structured legal text
def generate_legal_text(facts, arguments, demand, legal_consequence):
    print("📝 Generating legal text...")  # Debugging
    formatted_args = "\n".join([f"{i+1}. {arg}" for i, arg in enumerate(arguments) if arg])
    
    # Ensuring professional phrasing
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
        print("✅ Legal text generated successfully!")
        return response[0]["generated_text"]
    except Exception as e:
        print(f"❌ Error in text generation: {e}")
        return "Error generating legal text."

# Function to generate PDF
def generate_demand_letter(ref_our, ref_your, defendant, subject, client, facts, arguments, demand, legal_consequence, deadline, advocate):
    print("📄 Generating PDF document...")  # Debugging
    legal_text = generate_legal_text(facts, arguments, demand, legal_consequence)
    today_date = datetime.datetime.now().strftime("%d %B %Y")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Helper function to encode text safely
    def safe_text(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    # Header
    pdf.cell(200, 10, txt=safe_text("DEMAND LETTER"), ln=True, align='C')
    pdf.ln(10)

    # References & Date
    pdf.cell(200, 10, txt=safe_text(f"Our Ref: {ref_our}     Your Ref: {ref_your}"), ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=safe_text(f"Date: {today_date}"), ln=True)
    pdf.ln(10)

    # Defendant Address
    pdf.multi_cell(0, 10, txt=safe_text(defendant))
    pdf.ln(5)

    # Subject
    pdf.cell(200, 10, txt=safe_text(f"Subject: {subject}"), ln=True)
    pdf.ln(10)

    # Introduction
    pdf.multi_cell(0, 10, txt=safe_text(f"Dear {defendant},\n\nWe represent {client} regarding {subject}. Below are the details of this claim:"))
    pdf.ln(5)

    # Facts
    pdf.multi_cell(0, 10, txt=safe_text(f"**Background:**\n{facts}"))
    pdf.ln(5)

    # Legal Basis (Dynamic Arguments)
    if arguments:
        pdf.multi_cell(0, 10, txt=safe_text("**Legal Basis:**"))
        for i, arg in enumerate(arguments):
            if arg:
                pdf.multi_cell(0, 10, txt=safe_text(f"{i+1}. {arg}"))
        pdf.ln(5)

    # AI-generated legal text
    pdf.multi_cell(0, 10, txt=safe_text("**Legal Interpretation:**"))
    pdf.multi_cell(0, 10, txt=safe_text(legal_text))
    pdf.ln(5)

    # Demand
    pdf.multi_cell(0, 10, txt=safe_text(f"**Demand:**\n{demand}"))
    pdf.ln(5)

    # Legal Consequences
    pdf.multi_cell(0, 10, txt=safe_text(f"**Legal Consequences:**\n{legal_consequence}"))
    pdf.ln(5)

    # Deadline
    pdf.multi_cell(0, 10, txt=safe_text(f"Compliance with this demand is expected within {deadline} days. Failure to do so will result in immediate legal action."))
    pdf.ln(10)

    # Advocate Signature
    pdf.multi_cell(0, 10, txt=safe_text(f"**Sincerely,**\n\n{advocate}"))

    # Save PDF
    pdf_path = "demand_letter.pdf"
    pdf.output(pdf_path)

    print("✅ PDF Generated Successfully!")  # Debugging
    return pdf_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📜 AI-Powered Demand Letter Generator")

    ref_our = gr.Textbox(label="Our Reference")
    ref_your = gr.Textbox(label="Your Reference")
    defendant = gr.Textbox(label="Defendant's Name & Address")
    subject = gr.Textbox(label="Subject Matter")
    client = gr.Textbox(label="Client Name")
    facts = gr.Textbox(label="Facts of the Case", lines=3)
    
    # Flexible Argument Inputs
    arguments = gr.Textbox(label="Enter Arguments (Separate each with a new line)", lines=5)
    
    demand = gr.Textbox(label="Demand/Remedy")
    legal_consequence = gr.Textbox(label="Legal Consequences")
    deadline = gr.Number(label="Time for Compliance (Days)", value=7)
    advocate = gr.Textbox(label="Advocate's Name & Signature")
    
    generate_btn = gr.Button("Generate Demand Letter")
    output_pdf = gr.File(label="Download Demand Letter")
    
    status = gr.Textbox(label="Processing Status", interactive=True)
    
    def process_and_monitor(ref_our, ref_your, defendant, subject, client, facts, arguments, demand, legal_consequence, deadline, advocate):
        try:
            args_list = [arg.strip() for arg in arguments.split("\n") if arg.strip()]  # Split arguments into list
            status_text = "Generating legal text... 🚀"
            pdf_path = generate_demand_letter(ref_our, ref_your, defendant, subject, client, facts, args_list, demand, legal_consequence, deadline, advocate)
            status_text = "✅ PDF Generated Successfully! Download below."
        except Exception as e:
            status_text = f"❌ Error: {e}"
            pdf_path = None

        print("🎯 Process completed!")  # Debugging
        return pdf_path, status_text  

    generate_btn.click(
        process_and_monitor, 
        inputs=[ref_our, ref_your, defendant, subject, client, facts, arguments, demand, legal_consequence, deadline, advocate], 
        outputs=[output_pdf, status]
    )

demo.launch()
