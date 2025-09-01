# AI-Based Resume Shortlisting System
This project uses AI to automatically match resumes with job descriptions.

## Features
- Upload multiple resumes (PDF format)
- Upload a job description
- Get a similarity score between resumes and the job description
- Built with Streamlit, Python, and Sentence Transformers
  
## Project Structure
resume-shortlisting/
│
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── resumes/            # Folder for resumes (PDFs)
└── .gitignore          # Git ignored files

## Installation & Setup

```bash
1. Clone the repository
   git clone https://github.com/yourusername/resume-shortlisting.git
   cd resume-shortlisting

2. Install dependencies
   pip install -r requirements.txt

3. Run the app
   streamlit run app.py
