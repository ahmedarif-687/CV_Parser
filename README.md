#NLP-Based Resume Parser

📌 Project Overview

This project is an NLP-based Resume Parser that processes multiple resumes, extracts key information, and predicts the category of the CV using Machine Learning (SVM Model). It leverages spaCy for Named Entity Recognition (NER) and runs on a Streamlit web application for an interactive user interface.

🚀 Features

Parses multiple resumes in PDF or DOCX format.

Extracts Name, CGPA, Education, and Skills using NLP techniques.

Predicts CV category (e.g., Data Science, Software Engineering, Finance) using an SVM model.

Provides an intuitive Streamlit UI for ease of use.

Outputs results in structured JSON or Excel format.

🛠️ Technologies Used

Python 🐍

spaCy (for Named Entity Recognition)

Scikit-learn (SVM Model)

Streamlit (for UI)

Pandas (for data handling)  

📊 How It Works

Upload multiple CVs through the Streamlit interface.

The system extracts key details using spaCy NER.

The SVM model predicts the resume category.

Results are displayed on the Streamlit dashboard and can be downloaded as JSON or Excel.

🔮 Future Enhancements

Improve model accuracy with fine-tuned transformer models.

Add support for more languages.

Enhance UI with better visualization tools.

📜 License

This project is open-source and available under the MIT License.

🙌 Acknowledgments

Thanks to spaCy, Streamlit, and Scikit-learn for their awesome tools!


