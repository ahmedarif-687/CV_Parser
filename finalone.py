import streamlit as st
import pandas as pd
import spacy
import re
import os
import pickle
from io import BytesIO
import os
import pdfplumber
import re
from docling.document_converter import DocumentConverter


class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        self.rf_model = pickle.load(open("rf_model.pkl", "rb"))
        self.tfidf = pickle.load(open("tfidf.pkl", "rb"))
        self.le = pickle.load(open("encoder.pkl", "rb"))


model_loader = ModelLoader()
rf_model = model_loader.rf_model
tfidf = model_loader.tfidf
le = model_loader.le

nlp = spacy.load("en_core_web_sm")

UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_resume(txt):
    clean_text = re.sub(r"http\S+", " ", txt)  
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", clean_text)  
    clean_text = re.sub(r"\s+", " ", clean_text).strip()  
    return clean_text

def extract_text_with_pdfplumber(pdf_path):
    """
    Extract text from a PDF using pdfplumber.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_using_docling(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    markdown_content = result.document.export_to_markdown()
    return markdown_content
    

def extract_skills_section(markdown_content):
    """
    Extracts the Skills section from a CV in markdown format.
    """
    # Split the markdown content into lines
    lines = markdown_content.split('\n')
    
    # Initialize variables to store the extracted skills section
    skills_section = []
    in_skills_section = False

    # Keywords to identify the Skills section (case-insensitive)
    skills_keywords = [
        "skills", "technical skills", "key skills", "core competencies",
        "proficiencies", "expertise"
    ]

    # Iterate through the lines to extract the Skills section
    for line in lines:
        # Check if the line indicates the start of the Skills section
       
        if line.lower().startswith("## ") and any(keyword in line.lower() for keyword in skills_keywords):
            in_skills_section = True
            continue  # Skip the heading line

        # If we're in the Skills section, capture the lines
        if in_skills_section:
            # Stop capturing if we encounter a new section (e.g., "## Professional Experience")
            if line.strip().startswith("## "):
                break
            skills_section.append(line)

    # Join the lines to form the final extracted Skills section
    skills = '\n'.join(skills_section).strip()

    return skills


def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not Found"

def extract_cgpa(text):
    cgpa_match = re.findall(r"CGPA[:\-\s]*([\d\.]+)", text)
    return cgpa_match[0] if cgpa_match else "Not Found in CV"

def predict_category(input_resume):
    cleaned_text = clean_resume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    predicted_category = rf_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]


def extract_education_section(text):
    """
    Extract the "Education" section from the extracted text.
    """
    lines = text.split('\n')
    education_section = []
    in_education_section = False

    # Keywords to identify the Education section (case-insensitive)
    education_keywords = [
        "education", "educational background", "academic qualifications",
        "academics", "qualifications", "degrees", "academic history"
    ]

    for line in lines:
        # Check for the start of the Education section (case-insensitive)
        if any(keyword in line.lower() for keyword in education_keywords):
            in_education_section = True
            continue  # Skip the heading line itself

        # If we're in the Education section, collect the lines
        if in_education_section:
            # Check for the end of the section (a new main heading or a blank line followed by a heading)
            if line.strip().startswith("#") or line.strip().lower() in ["professional experience", "work experience", "skills", "certifications", "awards and activities"]:
                break
            education_section.append(line)

    # Join the lines to form the final Education section
    education = '\n'.join(education_section).strip()
    return education
 
def filter_education_section(education_section):
    """
    Filter the Education section to include only lines or parts of lines with specific keywords.
    If no keywords are found, return the entire section.
    """
    # Regular expression to match degree keywords (case-insensitive)
    degree_pattern = re.compile(
        r"\b(bachelor|master|phd|degree|bs|ms|bsc|msc|mba|b\.tech|m\.tech|b\.e|m\.e|b\.a|m\.a|b\.com|m\.com|high school|diploma)\b",
        re.IGNORECASE
    )

    filtered_lines = []

    # Split each line into parts and check for degree keywords
    for line in education_section.split('\n'):
        # Split the line into parts based on common delimiters (e.g., comma, period, dash)
        parts = re.split(r"[,.-]", line)
        for part in parts:
            if degree_pattern.search(part):
                filtered_lines.append(part.strip())

    # If no lines match the keywords, return the entire section
    if not filtered_lines:
        return education_section

    # Join the filtered lines to form the final Education section
    filtered_education = '\n'.join(filtered_lines).strip()
    return filtered_education

def save_pdf_links_to_excel(data):
    df = pd.DataFrame(data)
    df["PDF Link"] = df["File Path"].apply(lambda x: f'=HYPERLINK("{x}", "Open PDF")')
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df[["Candidate Name", "Predicted Category", "CGPA", "Skills", "Education", "PDF Link"]].to_excel(
            writer, sheet_name="Resume Data", index=False
        )
        workbook = writer.book
        worksheet = writer.sheets["Resume Data"]
        hyperlink_format = workbook.add_format({"font_color": "blue", "underline": 1})
        worksheet.set_column("F:F", 30, hyperlink_format) 
    output.seek(0)
    return output

def main():
    st.set_page_config(page_title="Resume Processor", page_icon="📄", layout="wide")
    st.title("📄 Resume Processing & Classification")
    st.markdown("Upload multiple resumes (PDF) and extract **Name, Category, CGPA, Skills, Education**. The results will be saved in an **Excel file** with a clickable link to each PDF.")
    
    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        results = []

        for uploaded_file in uploaded_files:
            try:
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                ed_text=extract_text_with_pdfplumber(file_path)
                Docling_text= extract_using_docling(file_path)
                name = extract_name(ed_text)
                cgpa = extract_cgpa(ed_text)
                category = predict_category(ed_text)
                skills = extract_skills_section(Docling_text)
                textt = extract_education_section(ed_text)
                education= filter_education_section(textt)


                results.append({
                    "File Name": uploaded_file.name,
                    "File Path": os.path.abspath(file_path),
                    "Candidate Name": name,
                    "Predicted Category": category,
                    "CGPA": cgpa,
                    "Skills": skills,
                    "Education": education,
                })

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        df = pd.DataFrame(results)
        st.subheader("📊 Extracted Information")
        st.dataframe(df[["Candidate Name", "Predicted Category", "CGPA", "Skills", "Education"]])

        excel_file = save_pdf_links_to_excel(results)
        st.download_button(label="📥 Download Excel File", data=excel_file, file_name="resume_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
