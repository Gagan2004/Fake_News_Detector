import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os

# CONFIGURATION
MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 2
SUBSET_SIZE = 5000 

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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def main():
    print("Loading data...")
    df = pd.read_csv("merged_dataset.csv")
    df['text'] = df['text'].fillna('')
    
    if SUBSET_SIZE:
        print(f"Using subset of {SUBSET_SIZE} for manual training...")
        df = df.sample(SUBSET_SIZE, random_state=42).reset_index(drop=True)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['label']),
        y=df['label']
    ).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights).to(device)
    print(f"Using device: {device}")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = FakeNewsDataset(train_df.text.to_numpy(), train_df.label.to_numpy(), tokenizer, MAX_LENGTH)
    val_dataset = FakeNewsDataset(val_df.text.to_numpy(), val_df.label.to_numpy(), tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print("Starting manual fine-tuning...")
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        print(f"Epoch {epoch+1} - Loss: {np.mean(losses):.4f}, Val Acc: {correct/total:.4f}")

    print("Saving model...")
    model.save_pretrained("./sota_fake_news_model")
    tokenizer.save_pretrained("./sota_fake_news_model")
    print("Done!")

if __name__ == "__main__":
    main()
