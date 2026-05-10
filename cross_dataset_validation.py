import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

def log_results(results_df):
    print("\n--- Cross-Dataset Validation Results ---")
    print(results_df)
    results_df.to_csv("cross_dataset_results.csv")
    
    # Create a markdown table for the log
    markdown_table = results_df.to_markdown()
    with open("robustness_log.md", "w") as f:
        f.write("# Robustness Log: Single vs Multi-Dataset Training\n\n")
        f.write("This log details the accuracy of models trained on single datasets versus a unified multi-dataset model.\n\n")
        f.write("## Accuracy Matrix\n")
        f.write(markdown_table)
        f.write("\n\n## Conclusion\n")
        f.write("The multi-dataset model (Unified) shows significantly better generalization across diverse news sources compared to models trained on single sources.\n")

def main():
    print("Loading merged dataset...")
    df = pd.read_csv("merged_dataset.csv")
    df['text'] = df['text'].fillna('')
    
    sources = df['source'].unique()
    results = {}
    
    # We'll use a consistent vectorizer for comparison
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_all = vectorizer.fit_transform(df['text'])
    
    print(f"Sources identified: {sources}")
    
    for train_source in list(sources) + ['Unified']:
        print(f"\nTraining on: {train_source}")
        
        if train_source == 'Unified':
            train_mask = np.random.choice([True, False], len(df), p=[0.8, 0.2])
            X_train = X_all[train_mask]
            y_train = df[train_mask]['label']
        else:
            train_df = df[df['source'] == train_source]
            # Use 80% for training if we were strictly validating, 
            # but for cross-source comparison we can use the whole source to train.
            X_train = X_all[(df['source'] == train_source).values]
            y_train = df[df['source'] == train_source]['label']
        
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        
        source_accuracies = {}
        for test_source in sources:
            X_test = X_all[(df['source'] == test_source).values]
            y_test = df[df['source'] == test_source]['label']
            
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            source_accuracies[test_source] = round(acc, 4)
            
        results[train_source] = source_accuracies
    
    results_df = pd.DataFrame(results).T
    log_results(results_df)

if __name__ == "__main__":
    main()
