import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_description_vector], resume_vectors).flatten()

# ---- Custom Styling ----
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main-title {
            color: #ffffff;
            background-color: #4CAF50;
            padding: 15px;
            text-align: center;
            border-radius: 8px;
            font-size: 24px;
            font-weight: bold;
        }
        .section-title {
            color: #ffffff;
            background-color: #008CBA;
            padding: 10px;
            border-radius: 5px;
            font-size: 20px;
        }
        .stTextArea>label {
            font-weight: bold;
        }
        .stFileUploader>label {
            font-weight: bold;
        }
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Sidebar Styling ----
st.sidebar.image(
    "https://th.bing.com/th/id/OIP.04z6Jxlf-9h660O4NpeNIAHaDn?w=349&h=170&c=7&r=0&o=5&dpr=1.3&pid=1.7", 
    use_container_width=True
)
st.sidebar.markdown("### 🔍 AI Resume Screening System")
st.sidebar.info("Upload resumes and enter a job description to rank candidates efficiently.")

# ---- Main App ----
st.markdown('<div class="main-title">AI Resume Screening & Candidate Ranking System</div>', unsafe_allow_html=True)

# Job description input
st.markdown('<div class="section-title">📄 Job Description</div>', unsafe_allow_html=True)

# Initialize session state for job description
if "job_description" not in st.session_state:
    st.session_state.job_description = ""

# Define callback function to clear job description
def clear_job_description():
    st.session_state.job_description = ""

# Text area for job description with session state
job_description = st.text_area("Enter the job description", value=st.session_state.job_description, key="job_description")

# Clear button that properly resets the input field
st.button("Clear Job Description", on_click=clear_job_description)

# ---- File Uploader with Clear All Resumes Feature ----
st.markdown('<div class="section-title">📂 Upload Resumes</div>', unsafe_allow_html=True)

# Initialize session state for uploaded files and clear trigger
if "clear_files_trigger" not in st.session_state:
    st.session_state.clear_files_trigger = False

# File uploader with dynamic key to reset it
uploaded_files = st.file_uploader(
    "Upload PDF files", 
    type=["pdf"], 
    accept_multiple_files=True, 
    key=f"upload_files_{st.session_state.clear_files_trigger}"  # Dynamic key to force reset
)

# Store uploaded files in session state
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# Define callback function to clear uploaded resumes
def clear_uploaded_resumes():
    st.session_state.clear_files_trigger = not st.session_state.clear_files_trigger  # Change key to reset uploader
    st.session_state.uploaded_files = []  # Reset stored files
    st.rerun()  # Force UI update

# Display "Clear All Resumes" button only if resumes are uploaded
if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
    if st.button("Clear All Resumes"): 
        clear_uploaded_resumes()  # ✅ Call function properly

# ---- Resume Ranking Logic ----
if "uploaded_files" in st.session_state and st.session_state.uploaded_files and job_description:
    st.markdown('<div class="section-title">📊 Ranking Resumes</div>', unsafe_allow_html=True)
    
    resumes = []
    for file in st.session_state.uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Create DataFrame with resume names and scores
    results = pd.DataFrame({"Resume": [file.name for file in st.session_state.uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    # Format score values to 4 decimal places
    results["Score"] = results["Score"].apply(lambda x: f"{x:.4f}")

    # Display the results
    st.dataframe(results, use_container_width=True)