import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
import evaluate  # Ensure 'evaluate' is installed
import os

def prepare_dataset(csv_file, label_to_id):
    """
    Load and prepare the dataset for NER training.
    Converts string labels to integer IDs before tokenization.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert string representation of lists to actual lists
    df['Labels'] = df['Labels'].apply(lambda x: eval(x))

    # Encode labels to integers using the label_to_id mapping
    df['labels'] = df['Labels'].apply(lambda labels: [label_to_id.get(label.strip(), -100) for label in labels])

    # Initialize tokenizer with padding and truncation
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', add_prefix_space=True)

    def tokenize_and_align_labels(examples):
        # Split the sentence into words
        words = [sentence.split() for sentence in examples['Sentence']]
        # Tokenize the words with padding and truncation
        tokenized_inputs = tokenizer(
            words,
            truncation=True,
            padding=True,       # Enable padding
            is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples['labels']):  # Access 'labels' field
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # Start of a new word
                    if word_idx < len(label):
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)  # Safety check
                else:
                    # Subsequent tokens in a word
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Convert DataFrame to Hugging Face Dataset, selecting only 'Sentence' and 'labels' columns
    dataset = Dataset.from_pandas(df[['Sentence', 'labels']])

    # Tokenize and align labels
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # Split into train and validation sets (90% train, 10% validation)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    return tokenized_dataset, tokenizer

def main():
    # Paths
    train_csv = 'synthetic_train.csv'  # Ensure this file exists in the same directory
    output_dir = 'medical-bert-symptom-ner'  # Directory to save the trained model

    # Define label mappings
    label_list = ['O', 'B-SYMPTOM', 'I-SYMPTOM']
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    # Load and prepare dataset with label encoding before tokenization
    tokenized_dataset, tokenizer = prepare_dataset(train_csv, label_to_id)

    # Initialize model
    model = AutoModelForTokenClassification.from_pretrained(
        'emilyalsentzer/Bio_ClinicalBERT',
        num_labels=len(label_list)
    )

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define training arguments with updated 'eval_strategy'
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # Initialize DataCollator for token classification to handle padding of labels
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Define compute_metrics inside main to access id_to_label
    def compute_metrics(p):
        """
        Compute evaluation metrics using seqeval.
        Converts integer predictions and labels back to string labels.
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Convert predictions and labels from IDs to label strings
        true_labels = []
        true_predictions = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_pred = []
            true_label = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue  # Ignore padding
                true_pred.append(id_to_label.get(pred_id, 'O'))
                true_label.append(id_to_label.get(label_id, 'O'))
            true_predictions.append(true_pred)
            true_labels.append(true_label)

        # Load the seqeval metric
        metric = evaluate.load("seqeval")

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Initialize Trainer with Early Stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")

    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to '{output_dir}' directory.")

if __name__ == "__main__":
    main()
