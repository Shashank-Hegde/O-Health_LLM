import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import process
import torch
import zipfile
import requests
import shutil
import spacy
import re
import tempfile

# Load SpaCy model for entity extraction
nlp = spacy.load('en_core_web_sm')

st.title("🩺 O-Health LLM")
st.write("""
    Enter the transcript with your symptoms, duration, age, location, gender, intensity etc below and we'll suggest possible diseases with associated probabilities.
""")

# URL to your model zip file hosted externally
MODEL_URL = "https://www.dropbox.com/scl/fi/pt0anz8mefta72rxjpyol/medical-bert-symptom-ner.zip?rlkey=ovtc18kbhw8njs3qwplcc76do&st=6y26kyl7&dl=1"

# Path to the model directory
model_dir = 'medical-bert-symptom-ner'  # Path where the model will be extracted

def download_and_unzip_model(model_url, model_dir):
    if not os.path.exists(model_dir):
        st.info("Downloading the model. Please wait...")
        try:
            with st.spinner('Downloading model...'):
                # Use a temporary file to store the downloaded zip
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                    response = requests.get(model_url, stream=True)
                    if response.status_code == 200:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                tmp_file.write(chunk)
                        tmp_zip_path = tmp_file.name
                    else:
                        st.error("Failed to download the model. Please check the URL.")
                        st.stop()

            # Unzip the model
            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            st.success("Model downloaded and extracted successfully.")

        except zipfile.BadZipFile:
            st.error("Downloaded file is not a valid zip file.")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred during download or extraction: {e}")
            st.stop()
        finally:
            # Remove the temporary zip file if it exists
            if 'tmp_zip_path' in locals() and os.path.exists(tmp_zip_path):
                try:
                    os.remove(tmp_zip_path)
                except Exception as e:
                    st.warning(f"Could not remove temporary file: {e}")

# Download and unzip the model if it doesn't exist
download_and_unzip_model(MODEL_URL, model_dir)

# Check if the model directory exists after extraction
if not os.path.exists(model_dir):
    st.error(f"Model directory '{model_dir}' not found after extraction.")
    st.stop()

# Load the tokenizer and model using caching
@st.cache_resource
def load_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)

ner_pipeline = load_ner_pipeline()
st.sidebar.success("NER model loaded successfully!")

# Load 'disease_symptom_mapping.csv'
if not os.path.exists("disease_symptom_mapping.csv"):
    st.error("'disease_symptom_mapping.csv' not found in the current directory.")
    st.stop()
df_disease_symptom = pd.read_csv("disease_symptom_mapping.csv")

# Prepare a list of known symptoms
known_symptoms = df_disease_symptom['SymptomName'].unique()

# Function to match extracted symptoms with known symptoms
def match_symptoms(extracted_symptoms):
    matched_symptoms = set()
    for symptom in extracted_symptoms:
        match = process.extractOne(symptom, known_symptoms, score_cutoff=80)
        if match:
            matched_symptoms.add(match[0])
    return matched_symptoms

# Function to extract additional entities
def extract_additional_entities(text):
    doc = nlp(text)
    age = None
    gender = None
    location = None
    duration = None
    medications = []

    # Extract entities using SpaCy
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            location = ent.text

    # Extract age
    age_patterns = [
        r'(\b\d{1,2}\b)\s*(years old|year old|y/o|yo|yrs old)',
        r'age\s*(\b\d{1,2}\b)'
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            break

    # Extract gender
    gender_patterns = [
        r'\b(male|female|man|woman|boy|girl)\b'
    ]
    for pattern in gender_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            gender = match.group(1).lower()
            if gender in ['man', 'boy']:
                gender = 'male'
            elif gender in ['woman', 'girl']:
                gender = 'female'
            break

    # Extract duration
    duration_patterns = [
        r'since\s*(\d+\s*(days?|weeks?|months?|years?))',
        r'for\s*(\d+\s*(days?|weeks?|months?|years?))',
        r'(\d+\s*(days?|weeks?|months?|years?))\s*(ago|back)'
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            duration = match.group(1)
            break

    # Extract medications
    # For simplicity, we'll assume that medications are mentioned after phrases like 'taking' or 'taken'
    medication_patterns = [
        r'(taking|taken|take)\s+([\w\s]+)'
    ]
    for pattern in medication_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            meds = match[1].strip()
            # Remove common words that are not medications
            stopwords = ['and', 'or', 'but', 'with', 'without']
            meds = ' '.join([word for word in meds.split() if word.lower() not in stopwords])
            medications.append(meds.title())

    return {
        'age': age,
        'gender': gender,
        'location': location,
        'duration': duration,
        'medications': medications
    }

# User input
user_input = st.text_area("📋 Enter your symptoms and additional information:", height=150,
                          placeholder="e.g., I am having fever since 2 days with cold and cough and nausea and high temperature. I have taken ibuprofen and have been sleepless. I am 33 years old male and I am from Mizoram near the mountains")

# Diagnose button
if st.button("Diagnose"):
    if user_input.strip() == "":
        st.warning("Please enter your symptoms for diagnosis.")
    else:
        with st.spinner("Analyzing your input..."):
            # Extract symptoms using NER pipeline
            entities = ner_pipeline(user_input)
            if not entities:
                st.error("No symptoms detected. Please try again.")
            else:
                # Extract unique symptoms
                extracted_symptoms = set([entity['word'].title() for entity in entities])
                symptoms = match_symptoms(extracted_symptoms)
                if not symptoms:
                    st.error("No matching symptoms found in our database.")
                else:
                    # Extract additional entities
                    additional_info = extract_additional_entities(user_input)

                    # Display extracted information
                    st.subheader("📝 Extracted Information:")
                    st.write(f"**Symptoms:** {', '.join(symptoms)}")
                    if additional_info['duration']:
                        st.write(f"**Duration:** {additional_info['duration']}")
                    if additional_info['age']:
                        st.write(f"**Age:** {additional_info['age']} years old")
                    if additional_info['gender']:
                        st.write(f"**Gender:** {additional_info['gender'].title()}")
                    if additional_info['location']:
                        st.write(f"**Location:** {additional_info['location']}")
                    if additional_info['medications']:
                        st.write(f"**Medications Taken:** {', '.join(additional_info['medications'])}")

                    # Create disease-symptom mapping
                    disease_symptom_map = df_disease_symptom.groupby('DiseaseName')['SymptomName'].apply(set).to_dict()

                    # Assume prior probabilities are equal for all diseases
                    disease_prior = {disease: 1 / len(disease_symptom_map) for disease in disease_symptom_map}

                    # Adjust priors based on age, gender, and location (simplified example)
                    for disease in disease_prior:
                        # Example adjustments (in reality, use actual data)
                        if additional_info['age']:
                            if disease in ['Chickenpox', 'Measles'] and additional_info['age'] > 12:
                                disease_prior[disease] *= 0.5  # Less likely in adults
                        if additional_info['gender']:
                            if disease == 'Prostate Cancer' and additional_info['gender'] == 'female':
                                disease_prior[disease] = 0  # Females do not get prostate cancer
                        if additional_info['location']:
                            if disease == 'Altitude Sickness' and 'mountain' in additional_info['location'].lower():
                                disease_prior[disease] *= 2  # More likely in mountains

                    # Calculate likelihoods and posterior probabilities
                    disease_scores = {}
                    for disease, symptoms_set in disease_symptom_map.items():
                        matched = symptoms.intersection(symptoms_set)
                        total_symptoms = len(symptoms_set)
                        if total_symptoms == 0:
                            continue
                        # Simple likelihood estimation
                        likelihood = len(matched) / total_symptoms
                        # Posterior probability proportional to likelihood * prior
                        posterior = likelihood * disease_prior[disease]
                        disease_scores[disease] = posterior

                    if disease_scores:
                        # Remove diseases with zero probability
                        disease_scores = {k: v for k, v in disease_scores.items() if v > 0}

                        if not disease_scores:
                            st.info("No probable diseases found based on the entered symptoms and information.")
                        else:
                            # Normalize the probabilities
                            total = sum(disease_scores.values())
                            for disease in disease_scores:
                                disease_scores[disease] = round((disease_scores[disease] / total) * 100, 2)
                            # Sort diseases by probability
                            sorted_diseases = dict(sorted(disease_scores.items(), key=lambda item: item[1], reverse=True))
                            # Display results
                            st.subheader("🩺 Probable Diseases:")
                            for disease, prob in sorted_diseases.items():
                                st.write(f"**{disease}**: {prob}%")

                            # Plot bar chart
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.barplot(x=list(sorted_diseases.keys()), y=list(sorted_diseases.values()), ax=ax)
                            ax.set_xlabel("Disease")
                            ax.set_ylabel("Probability (%)")
                            ax.set_title("Probable Diseases")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                    else:
                        st.info("No probable diseases found based on the entered symptoms and information.")
