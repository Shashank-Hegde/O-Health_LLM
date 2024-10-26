# O-Health_LLM
O-Health LLM for symptom extraction from input transcript

# Project File
The system has 3 python files:
1. generate_synthetic_data.py generates SNOMED based symptoms and also sentences of symptoms (synthetic_train.csv) as a csv file for synthetic data to train. Input : disease.csv, symptom.csv and disease_symptom_mapping.csv
2. train_distilbert.py takes as input the generated synthetic data and trains it with DBio_ClicalBERT and generates a NER file of the output. We then create a zip file of the NER folder for compression.
3. streamlit_app.py takes as input the zip file of a trained model and creates a web app to interact with symptoms as input and outputs probable diseases with graphs.

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
5. zip -r distilbert-symptom-ner.zip ./ (Or zip file manually)
6. In BioBert39 environment run streamlit with python 3.9
   streamlit run streamlit_app.py

Hosting on Streamlit cloud:
Add all the files to GitHub and upload the zip file of step 6 to DropBox. Copy the URL in streamlit.py and add ‘dl=1’ in link at the end to enable directdownload
Note: all csv files must be updated in the github repo linked to streamlit

https://o-health-v2.streamlit.app

Input Example:
I am having fever since 2 days with cold and cough and nausea and high temperature. I have taken ibuprofen and have been sleepless. I am 33 years old male and I am from Mizoram near the mountains

Next Steps:
1. Quantization: Prune and quantize/Distill the model after ‘step 4’ with quantize_model.py and select the zip model in streamlit_app
2. Recognising intensity and emotion related inputs as parameters to define the severity and impact of the diseases.
   Example: The model should consider adverbs and the tone of the sentence for the input.
   "The patient is very very tired" : The words "very very" should signify high intensity and severity of the symptoms
   "I was sleepy since last week but not anymore" : The sentence has positives and negatives and should be considered.



