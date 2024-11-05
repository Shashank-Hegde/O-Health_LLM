import os
import io
import re
import datetime
import logging
import base64
import uuid
import zipfile
import requests
import pandas as pd
import torch
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from googletrans import Translator, LANGUAGES
from textblob import TextBlob
import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
import random

# -------------------- Environment Setup -------------------- #

# Securely access the OpenAI API key from Streamlit Secrets
# Ensure you have a `.streamlit/secrets.toml` file with the following content:
# OPENAI_API_KEY = "your-openai-api-key"
openai.api_key = st.secrets["OPENAI_API_KEY"]

if not openai.api_key:
    st.error("OpenAI API key not found. Please set it in the Streamlit Secrets.")
    st.stop()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Load BioBERT NER Model -------------------- #

# URL to your BioBERT NER model zip file hosted externally
BIOBERT_MODEL_URL = "https://www.dropbox.com/scl/fi/odjgcsy5i8ktmpbag6p33/medical-bert-symptom-ner.zip?rlkey=j4ekri3mp92341o0wq2plnro6&st=0ucut9k7&dl=1"

# Path to the BioBERT model directory
BIOBERT_MODEL_DIR = 'medical-bert-symptom-ner'  # Path where the model will be extracted

def download_and_unzip_biobert_model(model_url, model_dir):
    if not os.path.exists(model_dir):
        st.info("Downloading the BioBERT NER model. Please wait...")
        with st.spinner('Downloading BioBERT NER model...'):
            try:
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                with open('biobert_model.zip', 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            out_file.write(chunk)
                logger.info("BioBERT NER model downloaded successfully.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download the BioBERT NER model: {e}")
                logger.error(f"Failed to download the BioBERT NER model: {e}")
                st.stop()
        # Unzip the model
        try:
            with zipfile.ZipFile('biobert_model.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            st.success("BioBERT NER model downloaded and extracted successfully.")
            logger.info("BioBERT NER model extracted successfully.")
        except zipfile.BadZipFile as e:
            st.error("Downloaded BioBERT model file is not a valid zip file.")
            logger.error(f"Invalid zip file: {e}")
            st.stop()
        finally:
            # Remove the zip file if it exists
            if os.path.exists('biobert_model.zip'):
                try:
                    os.remove('biobert_model.zip')
                    logger.info("biobert_model.zip removed successfully.")
                except Exception as e:
                    st.warning(f"Could not remove biobert_model.zip: {e}")
                    logger.warning(f"Could not remove biobert_model.zip: {e}")

# Download and unzip the BioBERT model if it doesn't exist
download_and_unzip_biobert_model(BIOBERT_MODEL_URL, BIOBERT_MODEL_DIR)

# Check if the BioBERT model directory exists after extraction
if not os.path.exists(BIOBERT_MODEL_DIR):
    st.error(f"BioBERT model directory '{BIOBERT_MODEL_DIR}' not found after extraction.")
    logger.error(f"BioBERT model directory '{BIOBERT_MODEL_DIR}' not found after extraction.")
    st.stop()

# Load the tokenizer and model using caching
@st.cache_resource
def load_biobert_ner_pipeline():
    try:
        tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_DIR, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(BIOBERT_MODEL_DIR)
        device = 0 if torch.cuda.is_available() else -1
        ner_pipeline_model = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)
        logger.info("BioBERT NER pipeline loaded successfully.")
        return ner_pipeline_model
    except Exception as e:
        st.error(f"Failed to load BioBERT NER pipeline: {e}")
        logger.error(f"Failed to load BioBERT NER pipeline: {e}")
        st.stop()

ner_pipeline = load_biobert_ner_pipeline()
st.sidebar.success("BioBERT NER model loaded successfully!")

# -------------------- Load SpaCy Model -------------------- #

@st.cache_resource
def load_spacy_model():
    """
    Load the SpaCy model for additional NLP tasks.
    """
    try:
        nlp = spacy.load('en_core_web_sm')
        logger.info("SpaCy model 'en_core_web_sm' loaded successfully.")
        return nlp
    except OSError as e:
        st.error("SpaCy model 'en_core_web_sm' not found. Please install it using 'python -m spacy download en_core_web_sm'.")
        logger.error(f"SpaCy model loading error: {e}")
        st.stop()

nlp = load_spacy_model()

# -------------------- Load Disease-Symptom Mapping -------------------- #

@st.cache_resource
def load_disease_symptom_mapping():
    """
    Load the disease-symptom mapping from a CSV file.
    """
    if not os.path.exists("disease_symptom_mapping.csv"):
        st.error("'disease_symptom_mapping.csv' not found in the current directory.")
        logger.error("'disease_symptom_mapping.csv' not found.")
        st.stop()
    try:
        df = pd.read_csv("disease_symptom_mapping.csv")
        logger.info("'disease_symptom_mapping.csv' loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Failed to load 'disease_symptom_mapping.csv': {e}")
        logger.error(f"Failed to load 'disease_symptom_mapping.csv': {e}")
        st.stop()

df_disease_symptom = load_disease_symptom_mapping()

# Prepare a list of known symptoms
known_symptoms = df_disease_symptom['SymptomName'].unique()

# -------------------- Define Symptom and Medication Lists -------------------- #

# Expanded symptom list with more symptoms and variations
symptom_list = [
    'fever', 'cough', 'headache', 'nausea', 'abdominal pain', 'chills',
    'vomiting', 'diarrhea', 'fatigue', 'shortness of breath', 'sore throat',
    'runny nose', 'sneezing', 'rash', 'dizziness', 'weakness', 'loss of appetite',
    'muscle pain', 'joint pain', 'chest pain', 'back pain', 'constipation',
    'throat pain', 'cold', 'flu', 'breathlessness', 'stomach ache', 'migraine',
    'pain', 'ache', 'sore', 'burning', 'itching', 'swelling', 'infection',
    'inflammation', 'cramps', 'ulcers', 'bleeding', 'irritation', 'anxiety',
    'depression', 'insomnia', 'cancer', 'diabetes', 'hypertension', 'allergies',
    'weight loss', 'weight gain', 'hair loss', 'blurred vision', 'ear pain',
    'palpitations', 'urinary frequency', 'urinary urgency', 'numbness', 'tingling',
    'night sweats', 'dry mouth', 'excessive thirst', 'frequent urination',
    'acne', 'bruising', 'confusion', 'memory loss', 'hoarseness', 'wheezing',
    'itchy eyes', 'dry eyes', 'difficulty swallowing', 'difficulty sleeping',
    'restlessness', 'yellow skin', 'yellow eyes', 'bloating', 'gas', 'hiccups',
    'indigestion', 'heartburn', 'mouth sores', 'nosebleeds', 'ear ringing',
    'decreased appetite', 'increased appetite', 'feeling full quickly',
    'unusual sweating', 'dark urine', 'light-colored stools', 'blood in urine',
    'blood in stool', 'frequent infections', 'delayed healing',
    'high temperature', 'high blood pressure', 'low blood pressure',
    # Add more symptoms and their variations as needed
]

# Expanded medications list
medications_list = [
    "ibuprofen", "dolo650", "paracetamol", "aspirin", "acetaminophen",
    "amoxicillin", "antibiotic", "metformin", "lisinopril", "atorvastatin",
    "cetirizine", "azithromycin", "hydrochlorothiazide", "omeprazole",
    "sertraline", "naproxen", "albuterol", "simvastatin", "loratadine",
    "furosemide", "clopidogrel", "prednisone", "insulin", "levothyroxine",
    "gabapentin", "citalopram", "hydrocodone", "codeine", "tramadol",
    "doxycycline", "ciprofloxacin", "lorazepam", "diazepam", "clonazepam",
    "pantoprazole", "escitalopram", "rosuvastatin", "omeprazole", "esomeprazole",
    "tamsulosin", "atenolol", "amlodipine", "sildenafil", "tadalafil",
    "clindamycin", "metronidazole", "acetylsalicylic acid", "nifedipine",
    "warfarin", "heparin", "digoxin", "fexofenadine", "salbutamol",
    "montelukast", "levocetirizine", "betahistine", "melatonin", "zinc",
    "vitamin c", "vitamin d", "multivitamin", "antacid", "antidepressant",
    # Add more medications as needed
]

# Initialize the translator
translator = Translator()

def translate_to_english(text):
    """
    Translate text to English if it's not already in English.
    """
    try:
        detection = translator.detect(text)
        if detection.lang != 'en':
            translated = translator.translate(text, dest='en')
            logger.info(f"Translated '{text}' from {LANGUAGES.get(detection.lang, 'unknown')} to English: '{translated.text}'")
            return translated.text
        return text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  # Fallback to original text if translation fails

def translate_to_hindi(text):
    """
    Translate text to Hindi.
    """
    try:
        translated = translator.translate(text, dest='hi')
        logger.info(f"Translated to Hindi: '{translated.text}'")
        return translated.text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  # Fallback to original text if translation fails

def correct_spelling(text):
    """
    Corrects the spelling of the given text using TextBlob.
    """
    try:
        corrected_text = str(TextBlob(text).correct())
        logger.info(f"Spelling corrected: '{text}' -> '{corrected_text}'")
        return corrected_text
    except Exception as e:
        logger.error(f"Spell correction failed: {e}")
        return text  # Return original text if correction fails

# -------------------- Symptom Follow-Up Questions -------------------- #

# Define symptom-specific follow-up questions with associated symptoms
symptom_followup_questions = {
    "Fever": [
        {"hi": "क्या आपका बुखार लगातार है या बीच-बीच में आता है?", "en": "Is your fever constant or intermittent?", "category": "fever_type", "symptom": None},
        {"hi": "क्या आपको ठंड लग रही है?", "en": "Are you experiencing any chills?", "category": "chills", "symptom": "Chills"},
        {"hi": "क्या आपने कोई दवा ली है?", "en": "Have you taken any medication?", "category": "medications", "symptom": None},
        {"hi": "क्या आपको सिरदर्द है?", "en": "Are you experiencing headaches?", "category": "headache", "symptom": "Headache"},
        {"hi": "क्या आपको उल्टी जैसा महसूस हो रहा है?", "en": "Are you feeling nauseous?", "category": "nausea", "symptom": "Nausea"},
        {"hi": "क्या आपका तापमान सामान्य से अधिक है?", "en": "Is your temperature higher than normal?", "category": "high_temperature", "symptom": "High temperature"},
        {"hi": "क्या आपको रात में पसीना आता है?", "en": "Do you experience night sweats?", "category": "night_sweats", "symptom": "Night sweats"},
        {"hi": "क्या आपको भूख कम लग रही है?", "en": "Are you experiencing loss of appetite?", "category": "loss_of_appetite", "symptom": "Loss of appetite"},
    ],
    "Cough": [
        {"hi": "क्या आपकी खांसी सूखी है या बलगम के साथ?", "en": "Is your cough dry or with phlegm?", "category": "cough_type", "symptom": None},
        {"hi": "क्या आपके खांसी के साथ बुखार है?", "en": "Do you have a fever along with your cough?", "category": "fever", "symptom": "Fever"},
        {"hi": "क्या आपको सांस लेने में कठिनाई हो रही है?", "en": "Are you experiencing difficulty breathing?", "category": "breathing", "symptom": "Shortness of breath"},
        {"hi": "क्या आपकी खांसी रात में बढ़ जाती है?", "en": "Does your cough worsen at night?", "category": "time", "symptom": None},
        {"hi": "क्या आपको सीने में दर्द है?", "en": "Are you experiencing chest pain?", "category": "chest_pain", "symptom": "Chest pain"},
        {"hi": "क्या आपको गले में खराश है?", "en": "Do you have a sore throat?", "category": "sore_throat", "symptom": "Sore throat"},
        {"hi": "क्या आपकी आवाज़ बदल गई है?", "en": "Has your voice changed?", "category": "voice_change", "symptom": "Hoarseness"},
        {"hi": "क्या आपको सांस लेने में सीटी जैसी आवाज़ आती है?", "en": "Do you experience wheezing?", "category": "wheezing", "symptom": "Wheezing"},
    ],
    "Abdominal Pain": [
        {"hi": "क्या दर्द पेट के ऊपरी हिस्से में है?", "en": "Is the pain in the upper abdomen?", "category": "upper_abdomen", "symptom": None},
        {"hi": "क्या दर्द पेट के निचले हिस्से में है?", "en": "Is the pain in the lower abdomen?", "category": "lower_abdomen", "symptom": None},
        {"hi": "क्या आपको उल्टी या मिचली हो रही है?", "en": "Are you experiencing vomiting or nausea?", "category": "nausea_vomiting", "symptom": "Nausea"},
        {"hi": "क्या आपको दस्त या कब्ज है?", "en": "Do you have diarrhea or constipation?", "category": "bowel_changes", "symptom": "Diarrhea"},
        {"hi": "क्या आपको पेट फूलना महसूस हो रहा है?", "en": "Are you feeling bloated?", "category": "bloating", "symptom": "Bloating"},
        {"hi": "क्या आपको खाना खाने के बाद दर्द बढ़ जाता है?", "en": "Does the pain increase after eating?", "category": "pain_after_eating", "symptom": None},
        {"hi": "क्या आपको एसिडिटी या जलन महसूस हो रही है?", "en": "Are you experiencing acidity or burning sensation?", "category": "acidity", "symptom": "Heartburn"},
        {"hi": "क्या आपको पेशाब में कोई समस्या है?", "en": "Do you have any problems with urination?", "category": "urination_issues", "symptom": None},
    ],
    "Stomach Ache": [
        {"hi": "क्या दर्द पेट के बीच में है?", "en": "Is the pain in the middle of your stomach?", "category": "middle_abdomen", "symptom": None},
        {"hi": "क्या आपको खाना नहीं पच रहा है?", "en": "Are you experiencing indigestion?", "category": "indigestion", "symptom": "Indigestion"},
        {"hi": "क्या आपको बुखार के साथ पेट दर्द है?", "en": "Do you have fever along with stomach ache?", "category": "fever", "symptom": "Fever"},
        {"hi": "क्या आपको पेट में ऐंठन हो रही है?", "en": "Are you experiencing cramps?", "category": "cramps", "symptom": "Cramps"},
        # Add more questions as needed
    ],
    # Add more symptom categories as needed
}

# Additional general follow-up questions
additional_followup_questions = [
    {"hi": "आपकी उम्र क्या है?", "en": "What is your age?", "category": "age", "symptom": None},
    {"hi": "आपका लिंग क्या है?", "en": "What is your gender?", "category": "gender", "symptom": None},
    {"hi": "आप वर्तमान में कहां स्थित हैं?", "en": "Where are you currently located?", "category": "location", "symptom": None},
    {"hi": "लक्षण कब से हैं?", "en": "How long have you had these symptoms?", "category": "duration", "symptom": None},
    {"hi": "क्या आप कोई अन्य लक्षण महसूस कर रहे हैं?", "en": "Are you experiencing any other symptoms?", "category": "other_symptoms", "symptom": None}
]

# -------------------- Core Functions -------------------- #

def generate_audio(text: str, lang: str = 'en') -> bytes:
    """
    Generate speech audio from text using gTTS and return it as bytes.
    """
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        logger.info("Audio generated successfully.")
        return audio_bytes
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        st.error(f"Failed to generate audio: {e}")
        return None

def embed_audio_autoplay(audio_bytes: bytes):
    """
    Embed audio in Streamlit app with autoplay using HTML and JavaScript.
    Includes a fallback play button if autoplay is blocked.
    """
    if audio_bytes is None:
        return

    # Encode audio to base64
    audio_base64 = base64.b64encode(audio_bytes).decode()

    # HTML and JavaScript to autoplay the audio
    audio_html = f"""
    <audio id="audioPlayer" src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>
    <script>
        window.addEventListener('load', function() {{
            var audio = document.getElementById('audioPlayer');
            audio.play().then(() => {{
                console.log('Audio played successfully');
            }}).catch((error) => {{
                console.log('Autoplay was prevented:', error);
                // Display a play button if autoplay is blocked
                var playButton = document.createElement('button');
                playButton.innerHTML = 'Play Audio';
                playButton.style.fontSize = '16px';
                playButton.style.padding = '10px 20px';
                playButton.style.marginTop = '20px';
                playButton.onclick = function() {{
                    audio.play();
                }};
                document.body.appendChild(playButton);
            }});
        }});
    </script>
    """

    # Embed the HTML in Streamlit
    components.html(audio_html, height=0, width=0)

def save_audio_file(audio_bytes, file_extension):
    """
    Save the audio bytes to a file with a unique name.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"

    try:
        with open(file_name, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Audio saved as {file_name}")
        return file_name
    except Exception as e:
        st.error(f"Failed to save audio file: {e}")
        logger.error(f"Failed to save audio file: {e}")
        return None

def transcribe_audio(file_path):
    """
    Transcribe the audio file using OpenAI's Whisper API with translation to English.
    """
    try:
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.translate("whisper-1", audio_file)
            transcribed_text = transcript.get("text", "").strip()
            logger.info(f"Audio transcription successful: {transcribed_text}")
            return transcribed_text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        logger.error(f"Transcription failed: {e}")
        return None
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Audio file {file_path} deleted successfully after transcription.")
            except Exception as e:
                st.warning(f"Could not delete audio file {file_path}: {e}")
                logger.warning(f"Could not delete audio file {file_path}: {e}")

def extract_symptoms(text):
    """
    Extract symptoms from the given text using BioBERT NER model and regex matching.
    """
    try:
        # Use BioBERT NER model to extract symptoms
        ner_results = ner_pipeline(text)
        extracted_symptoms = set()
        for entity in ner_results:
            if entity['entity_group'] == 'SYMPTOM':
                symptom = entity['word'].strip()
                # Ensure the symptom is in the known_symptoms list
                if symptom.title() in known_symptoms:
                    extracted_symptoms.add(symptom.title())
        logger.info(f"Extracted Symptoms using BioBERT: {extracted_symptoms}")

        # Also match against symptom list for any missed symptoms
        text_lower = text.lower()
        for symptom in symptom_list:
            symptom_lower = symptom.lower()
            if re.search(r'\b' + re.escape(symptom_lower) + r'\b', text_lower):
                extracted_symptoms.add(symptom.title())

        logger.info(f"Final Extracted Symptoms: {extracted_symptoms}")
        return extracted_symptoms
    except Exception as e:
        st.error(f"An error occurred during symptom extraction: {e}")
        logger.error(f"Symptom extraction error: {e}")
        return set()

def extract_additional_entities(text):
    """
    Extract additional entities like age, gender, location, duration, and medications from text.
    """
    doc = nlp(text)
    age = None
    gender = None
    location = None
    duration = None
    medications = []

    # Medications list
    tokens = [token.text.lower() for token in doc]
    for med in medications_list:
        if med.lower() in tokens:
            medications.append(med.title())
    medications = list(set(medications))  # remove duplicates

    # Extract age
    age_patterns = [
        r'(?i)\b(?:i am|i\'m|my age is|age)\s*(\d{1,3})\b',
        r'\b(\d{1,3})\s*(?:years old|year old|y/o|yo|yrs old|yrs|years)\b',
        r'\b(\d{1,3})\s*(?:male|female|man|woman|boy|girl)\b'
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text)
        if match:
            age_candidate = match.group(1)
            try:
                age = int(age_candidate)
                break
            except ValueError:
                continue

    # Extract gender
    gender_keywords = {'male', 'female', 'man', 'woman', 'boy', 'girl'}
    for token in doc:
        if token.text.lower() in gender_keywords:
            gender = token.text.lower()
            break

    # Extract location
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            location = ent.text
            break

    # Extract duration
    duration_patterns = [
        r'(?i)\b(?:for|since|from|past)\s+(?:the\s+)?(?:past\s+)?(\w+\s+(?:day|days|week|weeks|month|months|year|years))\b',
        r'\b(\w+\s+(?:day|days|week|weeks|month|months|year|years))\s+(?:ago|back)\b'
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, text)
        if match:
            duration_candidate = match.group(1)
            duration = duration_candidate.strip()
            break
    if not duration:
        # Try to extract DATE entities
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                duration = ent.text
                break

    # Ensure that duration is not misassigned from age
    if age and duration and str(age) in duration:
        duration = None

    logger.info(f"Extracted Entities: Age={age}, Gender={gender}, Location={location}, Duration={duration}, Medications={medications}")
    return {
        'age': age,
        'gender': gender,
        'location': location,
        'duration': duration,
        'medications': medications
    }

def determine_followup_questions(matched_symptoms, additional_info):
    """
    Determine the next set of follow-up questions based on matched symptoms and additional information.
    """
    followup_questions = []
    asked_categories = set()

    # Collect all possible questions for matched symptoms
    all_possible_questions = []
    for symptom in matched_symptoms:
        if symptom in symptom_followup_questions:
            symptom_questions = symptom_followup_questions[symptom]
            all_possible_questions.extend(symptom_questions)

    # Add additional follow-up questions
    all_possible_questions.extend(additional_followup_questions)

    # Remove duplicate questions based on 'category'
    unique_questions = {}
    for q in all_possible_questions:
        category = q.get('category')
        if category not in unique_questions and category not in asked_categories:
            unique_questions[category] = q

    all_possible_questions = list(unique_questions.values())

    # Randomly select up to 5 questions
    num_questions_to_ask = min(len(all_possible_questions), 5)
    if num_questions_to_ask > 0:
        selected_questions = random.sample(all_possible_questions, num_questions_to_ask)
    else:
        selected_questions = []

    # Build the follow-up questions list
    for q in selected_questions:
        category = q.get('category')
        if category not in asked_categories:
            # Skip questions if the information is already provided
            if category in additional_info and additional_info.get(category):
                continue
            followup_questions.append(q)
            asked_categories.add(category)
        if len(followup_questions) >= 5:
            break

    logger.info(f"Determined Follow-Up Questions: {followup_questions}")
    return followup_questions

def extract_all_symptoms(conversation_history):
    """
    Extract all symptoms and additional information from the conversation history.
    """
    matched_symptoms = set()
    additional_info = {
        'age': None,
        'gender': None,
        'location': None,
        'duration': None,
        'medications': []
    }

    affirmative_responses = {'yes', 'yeah', 'yep', 'yup', 'sure', 'of course', 'definitely', 'haan', 'ha'}
    negative_responses = {'no', 'nah', 'nope', 'not really', 'don\'t', 'nahi'}

    for entry in conversation_history:
        if 'user' in entry:
            user_text = entry['user']
            symptoms = extract_symptoms(user_text)
            matched_symptoms.update(symptoms)
            info = extract_additional_entities(user_text)
            for key in additional_info:
                if key in info and info[key]:
                    if isinstance(info[key], list):
                        additional_info[key].extend(info[key])
                        additional_info[key] = list(set(additional_info[key]))
                    else:
                        additional_info[key] = info[key]
        if 'followup_question_en' in entry:
            response_text = entry['response']
            symptoms = extract_symptoms(response_text)
            matched_symptoms.update(symptoms)
            info = extract_additional_entities(response_text)
            for key in additional_info:
                if key in info and info[key]:
                    if isinstance(info[key], list):
                        additional_info[key].extend(info[key])
                        additional_info[key] = list(set(additional_info[key]))
                    else:
                        additional_info[key] = info[key]

            # Infer symptoms from Yes/No answers to symptom-related questions
            question_entry = entry
            question_text = question_entry['followup_question_en']
            response_text_lower = response_text.strip().lower()

            # Check if response is affirmative
            is_affirmative = any(word in response_text_lower for word in affirmative_responses)

            # Get the 'symptom' associated with the question
            symptom = None
            for symptom_category in symptom_followup_questions.values():
                for q in symptom_category:
                    if q['en'] == question_text:
                        symptom = q.get('symptom')
                        break
                if symptom:
                    break

            if symptom and is_affirmative:
                matched_symptoms.add(symptom)
            elif symptom and any(word in response_text_lower for word in negative_responses):
                # Optionally, you can handle negative responses if needed
                pass

    logger.info(f"Final Matched Symptoms: {matched_symptoms}")
    logger.info(f"Additional Information: {additional_info}")
    return matched_symptoms, additional_info

def extract_and_prepare_questions(conversation_history):
    """
    Extract symptoms and additional info from the conversation history and prepare follow-up questions.
    """
    matched_symptoms, additional_info = extract_all_symptoms(conversation_history)
    st.session_state.additional_info = additional_info  # Store in session state for access elsewhere
    followup_questions = determine_followup_questions(matched_symptoms, additional_info)
    st.session_state.matched_symptoms = matched_symptoms  # Store matched symptoms in session state
    return followup_questions

def map_symptoms_to_diseases(matched_symptoms, additional_info):
    """
    Map the matched symptoms to probable diseases based on the disease-symptom mapping.
    """
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
        matched = matched_symptoms.intersection(symptoms_set)
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
            return {}
        else:
            # Normalize the probabilities
            total = sum(disease_scores.values())
            for disease in disease_scores:
                disease_scores[disease] = round((disease_scores[disease] / total) * 100, 2)
            # Sort diseases by probability
            sorted_diseases = dict(sorted(disease_scores.items(), key=lambda item: item[1], reverse=True))
            return sorted_diseases
    else:
        return {}

def generate_report(conversation_history):
    """
    Generate the final diagnostic report based on the conversation history.
    """
    matched_symptoms, additional_info = extract_all_symptoms(conversation_history)
    st.subheader("📄 **Final Report:**")
    st.write("**Symptoms:**", ', '.join(matched_symptoms) if matched_symptoms else 'Not specified')
    if additional_info['age']:
        st.write(f"**Age:** {additional_info['age']} years old")
    if additional_info['gender']:
        st.write(f"**Gender:** {additional_info['gender'].title()}")
    if additional_info['location']:
        st.write(f"**Location:** {additional_info['location']}")
    if additional_info['duration']:
        st.write(f"**Duration of Symptoms:** {additional_info['duration']}")
    if additional_info['medications']:
        st.write(f"**Medications Taken:** {', '.join(additional_info['medications'])}")

    # Map symptoms to diseases
    probable_diseases = map_symptoms_to_diseases(matched_symptoms, additional_info)

    if probable_diseases:
        # Display results
        st.subheader("🩺 **Probable Diseases:**")
        for disease, prob in probable_diseases.items():
            st.write(f"**{disease}**: {prob}%")

        # Plot bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(probable_diseases.keys()), y=list(probable_diseases.values()), ax=ax)
        ax.set_xlabel("Disease")
        ax.set_ylabel("Probability (%)")
        ax.set_title("Probable Diseases")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    else:
        st.info("No probable diseases found based on the entered symptoms and information.")

    st.subheader("📝 **Transcript of Questions and Answers:**")
    question_count = 1
    for entry in conversation_history:
        if 'followup_question_en' in entry and 'response' in entry:
            st.write(f"**Question {question_count} (English):** {entry['followup_question_en']}")
            st.write(f"**Your Answer:** {entry['response']}")
            st.write("---")
            question_count += 1

def handle_yes_no_response(question, response):
    """
    Handles yes/no responses to follow-up questions to add or remove symptoms.

    Args:
        question (dict): The current follow-up question being asked.
        response (str): The user's response to the follow-up question.
    """
    affirmative_responses = {'yes', 'yeah', 'yep', 'yup', 'sure', 'of course', 'definitely', 'haan', 'ha'}
    negative_responses = {'no', 'nah', 'nope', 'not really', 'don\'t', 'nahi'}

    response_lower = response.strip().lower()
    is_affirmative = any(word in response_lower for word in affirmative_responses)
    is_negative = any(word in response_lower for word in negative_responses)

    # Initialize session_state.matched_symptoms if not already
    if 'matched_symptoms' not in st.session_state:
        st.session_state.matched_symptoms = set()

    if is_affirmative and question['symptom']:
        st.session_state.matched_symptoms.add(question['symptom'])
        logger.info(f"Added symptom '{question['symptom']}' based on affirmative response.")
        st.success(f"Added symptom: {question['symptom']}")
    elif is_negative and question['symptom']:
        if question['symptom'] in st.session_state.matched_symptoms:
            st.session_state.matched_symptoms.remove(question['symptom'])
            logger.info(f"Removed symptom '{question['symptom']}' based on negative response.")
            st.warning(f"Removed symptom: {question['symptom']}")
    else:
        logger.info("Response not recognized as yes/no or no associated symptom.")

# -------------------- Main Streamlit Application -------------------- #

def main():
    # Initialize session state variables
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0  # Start at welcome
        st.session_state.conversation_history = []
        st.session_state.report_generated = False
        st.session_state.followup_questions = []
        st.session_state.current_followup = 0
        st.session_state.additional_info = {
            'age': None,
            'gender': None,
            'location': None,
            'duration': None,
            'medications': []
        }
        st.session_state.matched_symptoms = set()
        st.session_state.symptoms_processed = False

    st.title("🩺 O-Health Diagnostic Tool")
    st.write("""
        Welcome to the O-Health Diagnostic Tool. You can either speak your symptoms in Hindi or type them in English to receive potential disease recommendations based on your inputs.
    """)

    # Step 0: Welcome Message
    if st.session_state.current_step == 0:
        # Generate the welcome audio in Hindi
        welcome_text = "ओ-हेल्थ में आपका स्वागत है। कृपया माइक्रोफ़ोन बटन दबाएं और अपने लक्षण बोलें।"
        audio_bytes = generate_audio(welcome_text, lang='hi')
        if audio_bytes:
            # Attempt to embed and autoplay the audio
            embed_audio_autoplay(audio_bytes)
        else:
            st.error("Failed to generate welcome audio.")

        # Display a welcome message
        st.write("### Hello! 👋")
        st.write("We're glad to have you here.")
        st.write("Please provide your symptoms to get started.")

        st.session_state.current_step = 1  # Proceed to the next step

    # Step 1: Initial Symptom Input
    if st.session_state.current_step == 1:
        st.header("🗣️ Please Press the Microphone Button and Speak Your Symptoms:")
        # Record audio
        audio_bytes = audio_recorder(key="voice_input_initial")
        if audio_bytes and not st.session_state.get('symptoms_processed'):
            st.audio(audio_bytes, format="audio/wav")
            file_name = save_audio_file(audio_bytes, "wav")
            if file_name:
                st.success("Audio recorded and saved successfully!")
                st.info("Transcribing your audio... Please wait.")
                transcribed_text = transcribe_audio(file_name)
                if transcribed_text:
                    # Detect and translate to English if necessary
                    translated_text = translate_to_english(transcribed_text)
                    # Correct spelling in the translated text
                    corrected_text = correct_spelling(translated_text)
                    st.subheader("📝 Transcribed Text (English):")
                    st.write(corrected_text)
                    st.session_state.conversation_history.append({
                        'user': corrected_text
                    })
                    st.session_state.followup_questions = extract_and_prepare_questions(st.session_state.conversation_history)
                    st.session_state.current_step = 2  # Proceed to follow-up questions
                    st.session_state.symptoms_processed = True
                    st.experimental_rerun()
                else:
                    st.error("Failed to transcribe the audio.")
            else:
                st.error("Failed to save the audio file.")
        else:
            st.write("Please record your symptoms using the microphone button above.")
        # Optionally, provide a fallback text input
        st.write("**Alternatively, you can type your symptoms below:**")
        user_input = st.text_area("Enter your symptoms here...")
        if st.button("Submit Symptoms"):
            if user_input.strip() == "":
                st.warning("Please enter your symptoms.")
            else:
                # Detect and translate to English if necessary
                translated_input = translate_to_english(user_input)
                # Correct spelling in the translated text
                corrected_input = correct_spelling(translated_input)
                st.subheader("📝 Your Input:")
                st.write(corrected_input)
                st.session_state.conversation_history.append({
                    'user': corrected_input
                })
                st.session_state.followup_questions = extract_and_prepare_questions(st.session_state.conversation_history)
                st.session_state.current_step = 2  # Proceed to follow-up questions
                st.session_state.symptoms_processed = True
                st.experimental_rerun()

    # Step 2: Follow-Up Questions
    if st.session_state.current_step == 2:
        total_questions = len(st.session_state.followup_questions)
        if total_questions == 0:
            st.info("No follow-up questions required based on your inputs.")
            st.session_state.current_step = 3  # Proceed to report
            st.experimental_rerun()

        if st.session_state.current_followup < total_questions:
            current_question = st.session_state.followup_questions[st.session_state.current_followup]
            question_number = st.session_state.current_followup + 1

            # Display the question
            st.subheader(f"🔍 Follow-Up Question {question_number} of {total_questions}:")
            st.write(f"**Hindi:** {current_question['hi']}")
            st.write(f"**English:** {current_question['en']}")

            # Generate the question audio in Hindi
            if not st.session_state.get(f'question_{st.session_state.current_followup}_played'):
                question_audio = generate_audio(current_question['hi'], lang='hi')
                if question_audio:
                    # Attempt to embed and autoplay the audio
                    embed_audio_autoplay(question_audio)
                    st.session_state[f'question_{st.session_state.current_followup}_played'] = True
                else:
                    st.error("Failed to generate question audio.")

            # Record the user's answer
            st.write("**Please record your answer using the microphone button below:**")
            response_audio_bytes = audio_recorder(key=f"voice_input_followup_{st.session_state.current_followup}")
            if response_audio_bytes and not st.session_state.get(f'answer_{st.session_state.current_followup}_processed'):
                st.audio(response_audio_bytes, format="audio/wav")
                response_file_name = save_audio_file(response_audio_bytes, "wav")
                if response_file_name:
                    st.success("Audio recorded and saved successfully!")
                    st.info("Transcribing your audio... Please wait.")
                    response_transcribed = transcribe_audio(response_file_name)
                    if response_transcribed:
                        # Detect and translate to English if necessary
                        translated_response = translate_to_english(response_transcribed)
                        # Correct spelling in the translated text
                        corrected_response = correct_spelling(translated_response)
                        st.subheader(f"📝 Response to Follow-Up Question {question_number} (English):")
                        st.write(corrected_response)
                        # Handle yes/no responses to add/remove symptoms
                        handle_yes_no_response(current_question, corrected_response)
                        # Extract any new symptoms from the response
                        extracted_new_symptoms = extract_symptoms(corrected_response)
                        if extracted_new_symptoms:
                            st.session_state.matched_symptoms.update(extracted_new_symptoms)
                            st.success(f"New symptoms detected and added: {', '.join(extracted_new_symptoms)}")
                        st.session_state.conversation_history.append({
                            'followup_question_en': current_question['en'],
                            'response': corrected_response
                        })
                        st.session_state[f'answer_{st.session_state.current_followup}_processed'] = True
                        st.session_state.current_followup += 1
                        st.experimental_rerun()
                    else:
                        st.error("Failed to transcribe your audio response.")
                else:
                    st.error("Failed to save the audio file.")
            else:
                st.write("Please record your answer using the microphone button above.")

            # Optionally, provide a fallback text input
            st.write("**Alternatively, you can type your answer below:**")
            answer_input = st.text_input("Enter your answer here...", key=f"answer_input_{st.session_state.current_followup}")
            if st.button("Submit Answer", key=f"submit_answer_{st.session_state.current_followup}"):
                if answer_input.strip() == "":
                    st.warning("Please enter your answer.")
                else:
                    # Detect and translate to English if necessary
                    translated_answer = translate_to_english(answer_input)
                    # Correct spelling in the translated text
                    corrected_answer = correct_spelling(translated_answer)
                    st.session_state.conversation_history.append({
                        'followup_question_en': current_question['en'],
                        'response': corrected_answer
                    })
                    # Handle yes/no responses to add/remove symptoms
                    handle_yes_no_response(current_question, corrected_answer)
                    # Extract any new symptoms from the response
                    extracted_new_symptoms = extract_symptoms(corrected_answer)
                    if extracted_new_symptoms:
                        st.session_state.matched_symptoms.update(extracted_new_symptoms)
                        st.success(f"New symptoms detected and added: {', '.join(extracted_new_symptoms)}")
                    st.session_state[f'answer_{st.session_state.current_followup}_processed'] = True
                    st.session_state.current_followup += 1
                    st.experimental_rerun()
        else:
            # All follow-up questions have been asked
            st.session_state.current_step = 3
            st.experimental_rerun()

    # Step 3: Generate Report
    if st.session_state.current_step == 3 and not st.session_state.report_generated:
        st.session_state.report_generated = True
        with st.spinner("Analyzing your information..."):
            generate_report(st.session_state.conversation_history)

    # Display conversation logs on the sidebar
    with st.sidebar:
        st.header("📝 Conversation Log")
        for idx, entry in enumerate(st.session_state.conversation_history):
            if 'user' in entry:
                st.write(f"**User Input:** {entry['user']}")
            if 'followup_question_en' in entry:
                st.write(f"**Question {idx+1}:** {entry['followup_question_en']}")
                st.write(f"**Answer:** {entry['response']}")
        # Display extracted information
        matched_symptoms, additional_info = extract_all_symptoms(st.session_state.conversation_history)
        st.write("**Extracted Information:**")
        st.write(f"**Symptoms:** {', '.join(matched_symptoms) if matched_symptoms else 'Not specified'}")
        if additional_info['age']:
            st.write(f"**Age:** {additional_info['age']} years old")
        if additional_info['gender']:
            st.write(f"**Gender:** {additional_info['gender'].title()}")
        if additional_info['location']:
            st.write(f"**Location:** {additional_info['location']}")
        if additional_info['duration']:
            st.write(f"**Duration:** {additional_info['duration']}")
        if additional_info['medications']:
            st.write(f"**Medications Taken:** {', '.join(additional_info['medications'])}")

if __name__ == "__main__":
    main()
