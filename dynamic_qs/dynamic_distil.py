# health_reason_identifier_app.py

import streamlit as st
import spacy
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import subprocess

# Load spaCy English model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        # If the model is not present, download it
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# Load the tokenizer and model
@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_llm_model()

# Initialize the text generation pipeline
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)

# List of symptoms to look for
symptom_list = [
    'cough', 'fever', 'nausea', 'stomach ache', 'constipation', 'muscle pain',
    'high temperature', 'high blood pressure', 'fatigue', 'headache', 'indigestion',
    'itching', 'pain', 'swelling', 'vomiting', 'cold'
]

# Synonyms for symptoms
symptom_synonyms = {
    'high temperature': 'fever',
    'stomach ache': 'abdominal pain',
    'cold': 'common cold',
    'muscle pain': 'myalgia',
    'high blood pressure': 'hypertension',
}

# Factors to look for in input
factor_list = [
    'exposure to rain', 'allergies', 'smoking', 'stress', 'recent travel',
    'contact with sick person', 'poor diet', 'lack of sleep', 'physical exertion',
    'exposure to cold weather', 'pollution', 'dust', 'pollen', 'pet dander'
]

# Function to normalize symptoms
def normalize_symptom(symptom):
    return symptom_synonyms.get(symptom, symptom)

# Function to extract age
def extract_age(text):
    age = re.search(r'(\d{1,3})\s*(year old|years old|yo)\b', text)
    if age:
        return int(age.group(1))
    return None

# Function to extract gender
def extract_gender(doc):
    genders = {'male': 'male', 'female': 'female', 'man': 'male', 'woman': 'female', 'girl': 'female', 'boy': 'male'}
    for token in doc:
        if token.text.lower() in genders:
            return genders[token.text.lower()]
    return None

# Function to extract duration
def extract_duration(text):
    duration = re.search(r'(for|since)\s*(\d+)\s*(day|days|week|weeks|month|months|year|years)', text)
    if duration:
        return f"{duration.group(2)} {duration.group(3)}"
    return None

# Function to extract medications
def extract_medications(doc):
    medications = []
    for ent in doc.ents:
        if ent.label_ == 'DRUG':
            medications.append(ent.text)
    return medications if medications else None

# Function to extract symptoms
def extract_symptoms(doc):
    symptoms = []
    for token in doc:
        token_text = token.text.lower()
        if token_text in symptom_list:
            symptoms.append(normalize_symptom(token_text))
    return list(set(symptoms))

# Function to extract factors
def extract_factors(text):
    factors = []
    for factor in factor_list:
        if factor in text.lower():
            factors.append(factor)
    # Additional checks for keywords
    if 'rain' in text.lower() or 'drenched' in text.lower() or 'wet' in text.lower():
        factors.append('exposure to rain')
    if 'allerg' in text.lower():
        factors.append('allergies')
    if 'stress' in text.lower():
        factors.append('stress')
    if 'travel' in text.lower():
        factors.append('recent travel')
    return list(set(factors))

# Function to generate dynamic questions using LLM
def generate_dynamic_questions(symptoms, provided_params, num_questions=5):
    questions = []
    known_info = ''
    if provided_params['age']:
        known_info += f" The patient is {provided_params['age']} years old."
    if provided_params['gender']:
        known_info += f" The patient is {provided_params['gender']}."
    if provided_params['duration']:
        known_info += f" The symptoms have been present for {provided_params['duration']}."
    if provided_params['medications']:
        meds = ', '.join(provided_params['medications'])
        known_info += f" The patient is taking {meds}."
    if provided_params['factors']:
        factors = ', '.join(provided_params['factors'])
        known_info += f" Relevant factors include: {factors}."

    for symptom in symptoms:
        prompt = f"As a medical assistant, generate a question to gather more information about the patient's {symptom}.{known_info} Avoid asking about already provided information."
        # Generate text using the model
        response = text_generator(prompt, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
        question = response[0]['generated_text'].strip().split('\n')[0]
        # Ensure it's a question
        if not question.endswith('?'):
            question += '?'
        if question not in questions:
            questions.append(question)
        if len(questions) >= num_questions:
            break

    # Add general questions if needed
    general_prompts = [
        "Is there any other information you'd like to share about your condition?",
        "Are there any activities or situations that make your symptoms better or worse?",
        "Have you experienced similar symptoms in the past?",
        "Do you have any other symptoms that we haven't discussed?"
    ]
    for prompt in general_prompts:
        if len(questions) >= num_questions:
            break
        if prompt not in questions:
            questions.append(prompt)

    return questions[:num_questions]

# Function to infer possible reasons
def infer_reasons(transcript, answers):
    combined_text = transcript + ' ' + ' '.join(answers)
    factors = extract_factors(combined_text)
    return list(set(factors))

# Streamlit App
def main():
    st.title("Health Symptom Analyzer")
    st.write("**Disclaimer:** This tool is for informational purposes only and does not provide medical advice. Please consult a healthcare professional for medical concerns.")

    # User input
    user_input = st.text_area("Please describe your symptoms:", height=150)
    if st.button("Analyze"):
        if user_input.strip() == '':
            st.warning("Please enter a description of your symptoms.")
            return

        doc = nlp(user_input)

        # Extract parameters
        age = extract_age(user_input)
        gender = extract_gender(doc)
        duration = extract_duration(user_input)
        medications = extract_medications(doc)
        symptoms = extract_symptoms(doc)
        factors = extract_factors(user_input)

        provided_params = {
            'age': age,
            'gender': gender,
            'duration': duration,
            'medications': medications,
            'factors': factors
        }

        if not symptoms:
            st.warning("No recognizable symptoms found in your input. Please include symptoms like fever, cough, headache, etc.")
            return

        # Generate dynamic questions
        with st.spinner("Generating questions..."):
            questions = generate_dynamic_questions(symptoms, provided_params)

        # Display questions and collect answers
        st.subheader("Additional Questions")
        answers = []
        for idx, question in enumerate(questions, 1):
            answer = st.text_input(f"{idx}. {question}")
            answers.append(answer)

        if st.button("Submit Answers"):
            # Infer possible reasons
            possible_reasons = infer_reasons(user_input, answers)

            # Present possible reasons to the user
            st.subheader("Possible Reasons for Your Symptoms")
            if possible_reasons:
                for reason in possible_reasons:
                    st.write(f"- {reason.capitalize()}")
            else:
                st.write("We couldn't determine specific reasons based on the provided information.")
                st.write("Please consider consulting a healthcare professional for further assistance.")

if __name__ == "__main__":
    main()
