import fitz
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import base64

# -----------------------------
# Helper Functions
# -----------------------------

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
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="AI Resume Shortlisting", layout="wide")

st.title("🎯 AI-Based Resume Shortlisting System")
st.markdown("""
This tool uses **Machine Learning** + **Sentence Transformers** to identify  
the top-matching candidates based on your Job Description.
""")

# Job Description Input
job_description = st.text_area("📝 Enter Job Description", height=150)

# Resume Upload
uploaded_files = st.file_uploader(
    "📂 Upload PDF Resumes",
    accept_multiple_files=True,
    type=["pdf"]
)

# -----------------------------
# Ranking Logic
# -----------------------------
if st.button("✅ Rank Candidates"):

    if not job_description or not uploaded_files:
        st.error("Please enter Job Description and upload resumes!")
        st.stop()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_emb = model.encode([job_description])

    results = []
    resume_contents = {}

    for file in uploaded_files:
        pdf_bytes = file.read()
        text = clean_text(extract_text_from_pdf_bytes(pdf_bytes))
        emb = model.encode([text])
        score = cosine_similarity(job_emb, emb)[0][0]

        results.append([file.name, score])
        resume_contents[file.name] = pdf_bytes  # store PDF for preview

    df = pd.DataFrame(results, columns=["Resume", "Similarity Score"])
    df = df.sort_values(by="Similarity Score", ascending=False)

    st.subheader("🏆 **Top Ranked Candidates**")
    st.dataframe(df.style.format({"Similarity Score": "{:.4f}"}))

    # -----------------------------
    # Click to View Resume Section
    # -----------------------------
    st.subheader("📄 Click a Resume to View Details")

    selected_resume = st.selectbox(
        "Choose a Resume to Preview",
        df["Resume"].tolist()
    )

    if selected_resume:
        st.write(f"### 📌 Showing Resume: **{selected_resume}**")

        pdf_bytes = resume_contents[selected_resume]

        # Show PDF in browser
        show_pdf(pdf_bytes)

        # Extract & show text
        extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
        st.write("### 📘 Extracted Resume Text (for matching):")
        st.text(extracted_text[:1500] + "\n\n...")  # Show first part only

    # Download CSV button
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Ranking CSV", csv, "resume_rankings.csv")

