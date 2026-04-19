import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

MODEL_PATH = "./fake_news_model_final"

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0][pred].item()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train first.")
        return

    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    test_examples = [
        "The sky is falling and aliens have landed in New York!", # Fake (hopefully)
        "The president signed a new bill into law today after months of debate.", # Real
        "Drinking bleach cures all diseases according to a viral video." # Fake
    ]

    print("\n--- Manual Testing ---")
    for ex in test_examples:
        pred, conf = predict(ex, model, tokenizer)
        label = "FAKE" if pred == 1 else "REAL"
        print(f"Text: {ex[:50]}...")
        print(f"Prediction: {label} (Confidence: {conf:.2f})\n")

if __name__ == "__main__":
    import os
    main()
