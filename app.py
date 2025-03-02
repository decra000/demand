import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import re

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

# Split into smaller meaningful chunks (Paragraphs instead of Sections)
paragraphs = re.split(r'\n{2,}', legal_text)  # Splitting by double newlines (paragraphs)

# Create embeddings for each paragraph
embeddings = model.encode(paragraphs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Streamlit UI
st.title("Kenya Companies Act Chatbot")
st.write("Ask a question, and I'll find the most relevant section from the Companies Act of Kenya.")

query = st.text_input("Enter your legal question:")
if query:
    query_embedding = model.encode([query])
    
    # Retrieve top 3 most relevant sections
    _, closest_matches = index.search(np.array(query_embedding), 3)  # Get 3 best matches
    
    relevant_sections = [paragraphs[i] for i in closest_matches[0]]  # Extract matching paragraphs
    
    # Generate a human-readable summary (Simple approach)
    summary = "Based on your query, the Companies Act mentions the following key points:\n\n"
    for i, section in enumerate(relevant_sections, 1):
        summary += f"{i}. {section[:200]}...\n\n"  # Show first 200 characters
    
    st.subheader("Plain English Summary:")
    st.write(summary)
    
    st.subheader("Relevant Legal Sections:")
    for section in relevant_sections:
        st.write(section)
