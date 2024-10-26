import torch
from transformers import DistilBertForTokenClassification
import torch.nn.utils.prune as prune

def prune_model(trained_model_dir, pruned_model_dir, prune_amount=0.2):
    # Load the trained model
    model = DistilBertForTokenClassification.from_pretrained(trained_model_dir)

    # Apply pruning to all linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_amount)
            prune.remove(module, 'weight')  # Make pruning permanent

    # Save the pruned model
    model.save_pretrained(pruned_model_dir)
    print(f"Pruned model saved to '{pruned_model_dir}' directory.")

if __name__ == "__main__":
    trained_model_dir = 'distilbert-symptom-ner'  # Directory of the trained model
    pruned_model_dir = 'distilbert-symptom-ner-pruned'  # Directory to save the pruned model
    prune_model(trained_model_dir, pruned_model_dir)
