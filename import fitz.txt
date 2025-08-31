import fitz
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Helper Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Streamlit UI ---
st.title("AI-Based Resume Shortlisting System")

job_description = st.text_area("Enter Job Description", "")

uploaded_files = st.file_uploader(
    "Upload PDF Resumes", accept_multiple_files=True, type=["pdf"]
)

if st.button("Rank Resumes") and job_description and uploaded_files:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_emb = model.encode([job_description])
    
    results = []
    for file in uploaded_files:
        pdf_bytes = file.read()
        resume_text = clean_text(extract_text_from_pdf_bytes(pdf_bytes))
        resume_emb = model.encode([resume_text])
        score = cosine_similarity(job_emb, resume_emb)[0][0]
        results.append([file.name, score])
    
    df = pd.DataFrame(results, columns=["Resume Name", "Similarity Score"])
    df = df.sort_values(by="Similarity Score", ascending=False)
    
    st.subheader("Top Ranked Resumes")
    st.dataframe(df)
    
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "resume_rankings.csv")
