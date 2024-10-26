import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
import os

def download_nltk_resource(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

# Ensure NLTK resources are downloaded
download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/wordnet')
download_nltk_resource('taggers/averaged_perceptron_tagger')
download_nltk_resource('corpora/omw-1.4')

def get_synonyms(word, pos_tag):
    synonyms = set()
    wordnet_pos = None
    if pos_tag.startswith('N'):
        wordnet_pos = wordnet.NOUN
    elif pos_tag.startswith('V'):
        wordnet_pos = wordnet.VERB
    elif pos_tag.startswith('J'):
        wordnet_pos = wordnet.ADJ
    elif pos_tag.startswith('R'):
        wordnet_pos = wordnet.ADV
    if wordnet_pos:
        for syn in wordnet.synsets(word, pos=wordnet_pos):
            for lm in syn.lemmas():
                synonym = lm.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
    return list(synonyms)

def paraphrase_symptoms_in_sentence(sentence, selected_symptoms):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    symptom_words = set()
    for symptom in selected_symptoms:
        symptom_tokens = nltk.word_tokenize(symptom)
        symptom_words.update(symptom_tokens)

    new_sentences = []
    for i in range(len(words)):
        word = words[i]
        pos = pos_tags[i][1]
        if word in symptom_words:
            synonyms = get_synonyms(word, pos)
            if synonyms:
                synonym = random.choice(synonyms)
                new_sentence = words.copy()
                new_sentence[i] = synonym
                new_sentences.append(' '.join(new_sentence))
    return new_sentences

def generate_synthetic_train_csv(disease_symptom_mapping_df, num_samples=5000):
    sentences = []
    labels = []

    disease_symptom_map = disease_symptom_mapping_df.groupby('DiseaseName')['SymptomName'].apply(list).to_dict()

    for _ in range(num_samples):
        # Randomly choose a disease
        disease = random.choice(list(disease_symptom_map.keys()))
        disease_symptoms = disease_symptom_map[disease]
        if not disease_symptoms:
            continue
        # Randomly select 3 to 7 symptoms for the sentence
        num_symptoms_in_sentence = random.randint(3, min(7, len(disease_symptoms)))
        selected_symptoms = random.sample(disease_symptoms, num_symptoms_in_sentence)

        # Create a sentence template
        sentence_templates = [
            f"I have been experiencing {', '.join(selected_symptoms[:-1])}, and {selected_symptoms[-1]} for the past few days.",
            f"Symptoms include {', '.join(selected_symptoms)}.",
            f"My symptoms are {', '.join(selected_symptoms)}.",
            f"Recently, I have had {', '.join(selected_symptoms)}.",
            f"I am suffering from {', '.join(selected_symptoms)}.",
            f"For the last week, I've noticed {', '.join(selected_symptoms)}.",
            f"Could {', '.join(selected_symptoms)} be related?",
            f"I'm dealing with {', '.join(selected_symptoms)} and it's getting worse.",
            f"Experiencing {', '.join(selected_symptoms)} has been tough.",
            f"{', '.join(selected_symptoms)} have been bothering me."
        ]
        sentence = random.choice(sentence_templates)

        # Paraphrase symptoms in the sentence
        paraphrases = paraphrase_symptoms_in_sentence(sentence, selected_symptoms)
        all_sentences = [sentence] + paraphrases
        sentence = random.choice(all_sentences)
        sentences.append(sentence)

        # Labeling for NER (BIO scheme)
        tokens = nltk.word_tokenize(sentence)
        sentence_labels = ['O'] * len(tokens)
        for symptom in selected_symptoms:
            symptom_tokens = nltk.word_tokenize(symptom)
            # Find the starting index of the symptom in tokens
            for i in range(len(tokens) - len(symptom_tokens) + 1):
                match = True
                for j in range(len(symptom_tokens)):
                    if tokens[i + j].lower() != symptom_tokens[j].lower():
                        match = False
                        break
                if match:
                    sentence_labels[i] = 'B-SYMPTOM'
                    for j in range(1, len(symptom_tokens)):
                        sentence_labels[i + j] = 'I-SYMPTOM'
                    break  # Move to the next symptom once found
        labels.append(sentence_labels)

    # Save synthetic_train.csv in the same directory
    df_train = pd.DataFrame({'Sentence': sentences, 'Labels': labels})
    df_train.to_csv("synthetic_train.csv", index=False)
    print("Generated 'synthetic_train.csv' in the current directory.")

# Main function
def main():
    # Load disease_symptom_mapping.csv
    if not os.path.exists('disease_symptom_mapping.csv'):
        print("Error: 'disease_symptom_mapping.csv' not found.")
        return

    disease_symptom_mapping_df = pd.read_csv('disease_symptom_mapping.csv')

    generate_synthetic_train_csv(disease_symptom_mapping_df, num_samples=5000)

if __name__ == "__main__":
    main()
