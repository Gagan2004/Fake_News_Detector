import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def main():
    print("Loading merged dataset...")
    df = pd.read_csv("merged_dataset.csv")
    df['text'] = df['text'].fillna('')
    
    X = df['text']
    y = df['label']
    sources = df['source']
    unique_sources = df['source'].unique()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []

    print(f"Starting 5-fold Cross-Validation...")
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Processing Fold {fold + 1} ---")
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        source_test = sources.iloc[test_index]
        
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train_vec, y_train)
        
        # Global Evaluation
        y_pred = model.predict(X_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        res = {
            'fold': fold + 1,
            'dataset': 'Global',
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': report['macro avg']['f1-score'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall']
        }
        all_results.append(res)
        
        # Per-Dataset Evaluation (on this fold's test set)
        for src in unique_sources:
            mask = (source_test == src)
            if mask.any():
                X_src_test = X_test_vec[mask.values]
                y_src_test = y_test.iloc[mask.values]
                
                y_src_pred = model.predict(X_src_test)
                src_report = classification_report(y_src_test, y_src_pred, output_dict=True)
                
                all_results.append({
                    'fold': fold + 1,
                    'dataset': src,
                    'accuracy': accuracy_score(y_src_test, y_src_pred),
                    'f1_macro': src_report['macro avg']['f1-score'],
                    'precision_macro': src_report['macro avg']['precision'],
                    'recall_macro': src_report['macro avg']['recall']
                })

    results_df = pd.DataFrame(all_results)
    
    # Calculate Averages
    avg_results = results_df.groupby('dataset').mean().drop(columns='fold').reset_index()
    avg_results['fold'] = 'Average'
    
    final_df = pd.concat([results_df, avg_results]).sort_values(by=['dataset', 'fold'])
    
    print("\nSummary of Averages:")
    print(avg_results)
    
    final_df.to_csv("baseline_cv_results.csv", index=False)
    print("\nFull results saved to baseline_cv_results.csv")

if __name__ == "__main__":
    main()
