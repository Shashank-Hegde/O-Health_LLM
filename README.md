# O-Health_LLM
O-Health LLM for symptom extraction from input transcript

# Project File
The system has 3 python files:
1. generate_synthetic_data.py generates SNOMED based symptoms and also sentences of symptoms (synthetic_train.csv) as a csv file for synthetic data to train. Input : disease.csv, symptom.csv and disease_symptom_mapping.csv
2. train_distilbert.py takes as input the generated synthetic data and trains it with DBio_ClicalBERT and generates a NER file of the output. We then create a zip file of the NER folder for compression.
3. streamlit_app2.py OR streamlit_voice.py takes as input the zip file of a trained model and creates a web app to interact with symptoms through voice in multi-language (streamlit_voice.py) or through text (streamlit_app2) as input, and outputs probable diseases with graphs.
We can interact through streamlit cloud to test the features on:

https://o-health-v3.streamlit.app for streamlit_app2 OR

https://o-health-v4.streamlit.ap for streamllit_voice.py

# Project Running Procedure
The 3 python files currently are not running in the same python version due to conflicts in library versions. So we create 2 virtual environments, one with python 3.9 to run generate_synthetic_data.py and streamline_app.py, and python 3.12 for train_distilbert.py

Create a virtual environment named 'BioBert39' with python 3.9 and ‘BioBert312’ with python 3.12
1. python3 -m venv BioBert39

   source BioBert39/bin/activate

   Similar for BioBert312

2. pip3 install -r requirements.txt
3. In BioBert39 run synthetic_data.py
    python3 generate_synthetic_data.py
4. In BioBert312 the next one-
   python3 train_distilbert.py
5. zip -r medical-bert-symptom-ner.zip ./ (Or zip file manually)
6. In BioBert39 environment run streamlit with python 3.9

   streamlit run streamlit_app2.py

Hosting on Streamlit cloud:
Add all the files to GitHub and upload the zip file of step 6 to DropBox. Copy the URL in streamlit.py and add ‘dl=1’ in link at the end to enable directdownload
Note: all csv files must be updated in the github repo linked to streamlit

Input Example:
I am having fever since 2 days with cold and cough and nausea and high temperature. I have taken ibuprofen and have been sleepless. I am 33 years old male and I am from Mizoram near the mountains.

Output:
Extracts the data from the input
Extracted Information:
Symptoms: Cough, Fever, High Temperature, Sleepiness, Nausea
Duration: since 2 days
Age: 33 years old
Gender: Male
Medications Taken: Ibuprofen

Next Steps:
1. Quantization: Prune and quantize/Distill the model after ‘step 4’ with quantize_model.py and select the zip model in streamlit_app
2. Recognising intensity and emotion related inputs as parameters to define the severity and impact of the diseases.
   Example: The model should consider adverbs and the tone of the sentence for the input.
   "The patient is very very tired" : The words "very very" should signify high intensity and severity of the symptoms
   "I was sleepy since last week but not anymore" : The sentence has positives and negatives and should be considered.
3. Making the program more efficient in extracting the information from the transcript. Currently it extracts symptoms, medications taken, age, gender, duration and location but we can extract more information with more efficiency.
4. Integrating the entire system onto 1 python version (either 3.9 or 3.12) for easy development
5. Creating a model or program for root cause analysis of the disease from the inputs of the symptoms
Example input "I have a fever since 2 days ever since I was drenched in the rain, and rashes on my skin since yesterday. I have peanut allergy and had some roasted peanuts and nutella yesterday"
Expected Output: Your fever might be caused due to the rain and rashes could be an allergic reaction to the peanuts
6. Increase and customize the questions for each symptom extracted to make the dynamic questioning more effective on streamlit_voice.py
7. Train the model with more diseases and symptom
