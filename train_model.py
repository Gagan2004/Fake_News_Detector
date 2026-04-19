import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# CONFIGURATION
MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
SUBSET_SIZE = 10000  # Set to None for full training. Highly recommended for CPU.

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions)
    }

def main():
    print(f"Loading dataset...")
    df = pd.read_csv("merged_dataset.csv")
    
    if SUBSET_SIZE:
        print(f"Sampling {SUBSET_SIZE} rows for CPU-friendly training...")
        # Balanced sampling if possible
        df_fake = df[df['label'] == 1]
        df_real = df[df['label'] == 0]
        
        n_each = SUBSET_SIZE // 2
        df_fake_sub = df_fake.sample(min(len(df_fake), n_each))
        df_real_sub = df_real.sample(min(len(df_real), n_each))
        
        df = pd.concat([df_fake_sub, df_real_sub]).sample(frac=1).reset_index(drop=True)

    print(f"Splitting data...")
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    print(f"Initializing Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    print(f"Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    print(f"Loading Model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="./fake_news_model",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=100,
        push_to_hub=False,
        no_cuda=not torch.cuda.is_available(), # Use CPU if CUDA not available
        report_to="none"
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print(f"Starting Training...")
    trainer.train()
    
    print(f"Saving final model...")
    trainer.save_model("./fake_news_model_final")
    print("Training complete! Model saved to ./fake_news_model_final")

if __name__ == "__main__":
    main()
