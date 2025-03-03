import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import re
from transformers import pipeline  # For summarization

# Load AI model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            extracted_text = page.extract_text()
            if extracted_text:
                text += f"\n\n=== Page {page_num + 1} ===\n\n"  # Add page numbers for debugging
                text += extracted_text + "\n\n"
    return text

# Load and process legal text
pdf_path = "CompaniesAct17of2015 (3).pdf"
legal_text = extract_text_from_pdf(pdf_path)

# Debugging: Show first 1000 characters of extracted text
st.subheader("Preview of Extracted Text:")
st.text(legal_text[:1000])  # Print first 1000 characters to check format

# Attempt to find section patterns in the text
section_pattern = r"(SECTION\s+\d+|\bPART\s+\w+|\bCHAPTER\s+\w+)"  # More flexible regex
sections = re.split(section_pattern, legal_text, flags=re.IGNORECASE)

# Check if sections were split properly
if len(sections) < 2:
    st.error("Error: Unable to split the text into valid sections. The document format may not match expected patterns.")
    st.stop()

# Reconstruct sections properly
structured_chunks = []
for i in range(1, len(sections), 2):
    section_title = sections[i].strip()
    section_body = sections[i + 1].strip() if i + 1 < len(sections) else ""
    structured_chunks.append(f"{section_title}\n{section_body}")

# Debugging: Show first few extracted sections
st.subheader("Extracted Sections Preview:")
for i in range(min(3, len(structured_chunks))):
    st.text(f"--- {structured_chunks[i][:500]}... ---")

# Ensure structured_chunks has valid data
if not structured_chunks:
    st.error("Error: No valid sections were created. The document formatting may need manual inspection.")
    st.stop()

# Create embeddings for each section
try:
    embeddings = model.encode(structured_chunks)
    if embeddings.size == 0:
        raise ValueError("Embedding array is empty.")
except Exception as e:
    st.error(f"Error in embedding generation: {str(e)}")
    st.stop()

# Create FAISS index
try:
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
except Exception as e:
    st.error(f"Error initializing FAISS index: {str(e)}")
    st.stop()

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
