import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import re
from transformers import pipeline  # For better summarization

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

# Split text into meaningful sections (Using "SECTION" as a delimiter)
sections = re.split(r"(SECTION \d+)", legal_text)  # Splitting based on section headings

# Reconstruct sections properly
structured_chunks = []
for i in range(1, len(sections), 2):
    section_title = sections[i].strip()
    section_body = sections[i + 1].strip() if i + 1 < len(sections) else ""
    structured_chunks.append(f"{section_title}\n{section_body}")

# Create embeddings for each section
embeddings = model.encode(structured_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit UI
st.title("Kenya Companies Act Chatbot")
st.write("Ask a question, and I'll find the most relevant section from the Companies Act of Kenya.")

query = st.text_input("Enter your legal question:")
if query:
    query_embedding = model.encode([query])
    
    # Retrieve top 3 most relevant sections
    _, closest_matches = index.search(np.array(query_embedding), 3)
    
    relevant_sections = [structured_chunks[i] for i in closest_matches[0]]

    # Generate a summary of the best-matching sections
    combined_text = " ".join(relevant_sections[:2])  # Use first 2 sections for summarization
    summary_text = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

    st.subheader("Plain English Summary:")
    st.write(summary_text)
    
    st.subheader("Relevant Legal Sections:")
    for section in relevant_sections:
        st.write(section)
