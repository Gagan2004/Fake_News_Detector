import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
import os

def main():
    print("Loading data for ensemble...")
    # Use a small test set for the ensemble demonstration
    df = pd.read_csv("merged_dataset.csv").sample(500, random_state=100).reset_index(drop=True)
    
    # 1. Logistic Model Predictions
    print("Getting Logistic Regression predictions...")
    if os.path.exists("models/fake_news_model.pkl"):
        lr_model = joblib.load("models/fake_news_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        lr_vec = vectorizer.transform(df['text'].fillna(''))
        lr_probs = lr_model.predict_proba(lr_vec)
        lr_preds = lr_model.predict(lr_vec)
    else:
        print("Logistic model not found. Using dummy predictions.")
        lr_probs = np.random.rand(len(df), 2)
        lr_preds = np.argmax(lr_probs, axis=1)

    # 2. SOTA (BERT) Model Predictions
    print("Getting SOTA (BERT) predictions...")
    if os.path.exists("./sota_fake_news_model"):
        tokenizer = AutoTokenizer.from_pretrained("./sota_fake_news_model")
        model = AutoModelForSequenceClassification.from_pretrained("./sota_fake_news_model")
        model.eval()
        
        bert_probs = []
        with torch.no_grad():
            for text in df['text'].fillna(''):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
                bert_probs.append(probs)
        bert_probs = np.array(bert_probs)
        bert_preds = np.argmax(bert_probs, axis=1)
    else:
        print("SOTA model not found. Using dummy predictions.")
        bert_probs = np.random.rand(len(df), 2)
        bert_preds = np.argmax(bert_probs, axis=1)

    # 3. Soft Voting Ensemble
    print("Calculating Ensemble (Soft Voting)...")
    ensemble_probs = (lr_probs + bert_probs) / 2
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    # 4. Evaluation
    y_true = df['label']
    print("\n--- Logistic Regression Results ---")
    print(classification_report(y_true, lr_preds))
    
    print("\n--- SOTA (BERT) Results ---")
    print(classification_report(y_true, bert_preds))
    
    print("\n--- Ensemble Results ---")
    print(classification_report(y_true, ensemble_preds))
    
    # Save predictions for McNemar test
    results_df = pd.DataFrame({
        'y_true': y_true,
        'lr_preds': lr_preds,
        'bert_preds': bert_preds,
        'ensemble_preds': ensemble_preds
    })
    results_df.to_csv("ensemble_predictions.csv", index=False)
    print("\nPredictions saved to ensemble_predictions.csv for statistical testing.")

if __name__ == "__main__":
    main()
