import fitz
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import base64

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

def show_pdf(pdf_bytes):
    """ Display PDF directly inside Streamlit """
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}"
        width="700" height="900" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- Streamlit UI ---
st.title("📄 AI-Based Resume Shortlisting System")

job_description = st.text_area("📝 Enter Job Description")

uploaded_files = st.file_uploader(
    "📤 Upload PDF Resumes", 
    accept_multiple_files=True, 
    type=["pdf"]
)

if st.button("✅ Rank Resumes") and job_description and uploaded_files:
    
    st.info("Processing resumes... please wait ⏳")
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_emb = model.encode([job_description])

    results = []
    file_bytes_map = {}  # store original files

    # Process each resume
    for file in uploaded_files:
        pdf_bytes = file.read()
        file_bytes_map[file.name] = pdf_bytes
        
        resume_text = clean_text(extract_text_from_pdf_bytes(pdf_bytes))
        resume_emb = model.encode([resume_text])
        score = cosine_similarity(job_emb, resume_emb)[0][0]

        results.append([file.name, score])

    # Ranking table
    df = pd.DataFrame(results, columns=["Resume Name", "Similarity Score"])
    df = df.sort_values(by="Similarity Score", ascending=False)

    st.subheader("🏆 Top Ranked Resumes")
    st.dataframe(df)

    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button("⬇ Download CSV", csv, "resume_rankings.csv")

    st.subheader("📂 Open Resume")

    # Create buttons for viewing resumes
    for name in df["Resume Name"]:
        if st.button(f"📄 Open {name}"):
            st.write(f"### Showing Resume: {name}")
            show_pdf(file_bytes_map[name])
