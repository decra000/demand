import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2

# Load AI model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    return text

# Load and process legal text
pdf_path = "CompaniesAct17of2015 (3).pdf"
legal_text = extract_text_from_pdf(pdf_path)
sections = legal_text.split("SECTION ")  # Splitting into sections

# Create embeddings for each section
embeddings = model.encode(sections)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Streamlit UI
st.title("Kenya Companies Act Chatbot")
st.write("Ask a question, and I'll find the most relevant section from the Companies Act of Kenya.")

query = st.text_input("Enter your legal question:")
if query:
    query_embedding = model.encode([query])
    _, closest_match = index.search(np.array(query_embedding), 1)
    st.subheader("Relevant Section:")
    st.write(sections[closest_match[0][0]])
