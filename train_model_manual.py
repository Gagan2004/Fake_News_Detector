import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
from tqdm import tqdm

# CONFIGURATION
MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 128  # Reduced for speed on CPU
BATCH_SIZE = 8
EPOCHS = 1 # Keep low for CPU demonstration
LEARNING_RATE = 2e-5
SUBSET_SIZE = 200 # Increase this for real training

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    print(f"Loading dataset...")
    df = pd.read_csv("merged_dataset.csv")
    
    if SUBSET_SIZE:
        print(f"Sampling {SUBSET_SIZE} rows for CPU training...")
        df = df.sample(min(len(df), SUBSET_SIZE)).reset_index(drop=True)

    print(f"Splitting data...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = FakeNewsDataset(
        texts=train_df.text.to_numpy(),
        labels=train_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LENGTH
    )
    
    val_dataset = FakeNewsDataset(
        texts=val_df.text.to_numpy(),
        labels=val_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Loading Model...")
    device = torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        # Validation
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                actuals.extend(labels.cpu().numpy())
        
        acc = accuracy_score(actuals, preds)
        f1 = f1_score(actuals, preds)
        print(f"Epoch {epoch+1} - Loss: {np.mean(losses):.4f}, Val Acc: {acc:.4f}, Val F1: {f1:.4f}")

    print(f"Saving model...")
    model.save_pretrained("./fake_news_model_final")
    tokenizer.save_pretrained("./fake_news_model_final")
    print("Done!")

if __name__ == "__main__":
    main()
