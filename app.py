import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import re
from transformers import pipeline

# Load AI model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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
st.write("Ask a question, and I'll find the most relevant legal sections.")

query = st.text_input("Enter your legal question:")
if query:
    query_embedding = model.encode([query])
    _, closest_matches = index.search(np.array(query_embedding), 3)  # Retrieve top 3 matches
    
    related_paragraphs = [paragraphs[i] for i in closest_matches[0]]

    # Combine for AI summarization
    combined_text = " ".join(related_paragraphs[:2])  # Use top 2 for summary

    # Prevent errors by handling empty or too-long input
    if combined_text.strip():
        MAX_INPUT_LENGTH = 1024
        if len(combined_text.split()) > MAX_INPUT_LENGTH:
            combined_text = " ".join(combined_text.split()[:MAX_INPUT_LENGTH])  # Truncate

        summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    else:
        summary = "No relevant information found in the Companies Act for your query."

    st.subheader("AI-Generated Explanation:")
    st.write(summary)

    st.subheader("Relevant Legal Sections:")
    for para in related_paragraphs:
        st.write(para)
