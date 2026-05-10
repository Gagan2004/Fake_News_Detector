import pandas as pd
import numpy as np
from scipy.stats import chi2
import os

def mcnemar_test(y_true, pred1, pred2):
    """
    Computes McNemar's test for two sets of predictions.
    Contingency table:
                Model 2 Correct | Model 2 Wrong
    Model 1 Correct    a        |       b
    Model 1 Wrong      c        |       d
    """
    a = np.sum((pred1 == y_true) & (pred2 == y_true))
    b = np.sum((pred1 == y_true) & (pred2 != y_true))
    c = np.sum((pred1 != y_true) & (pred2 == y_true))
    d = np.sum((pred1 != y_true) & (pred2 != y_true))
    
    # McNemar's test statistic with continuity correction
    stat = ((abs(b - c) - 1)**2) / (b + c) if (b + c) > 0 else 0
    p_value = 1 - chi2.cdf(stat, 1)
    
    return stat, p_value

def main():
    print("Starting Scientific Evaluation...")
    
    # 1. Generate Tables from CV results
    if os.path.exists("baseline_cv_results.csv"):
        cv_df = pd.read_csv("baseline_cv_results.csv")
        avg_table = cv_df[cv_df['fold'] == 'Average'].drop(columns='fold')
        print("\n--- Table 1: Per-Dataset Average Metrics (Baseline) ---")
        print(avg_table.to_markdown(index=False))
        
        # Cross-Dataset Matrix (Simplified representation)
        print("\n--- Table 2: Cross-Dataset Consistency ---")
        print("Dataset-specific F1 scores across 5 folds:")
        pivot = cv_df[cv_df['fold'] != 'Average'].pivot(index='fold', columns='dataset', values='f1_macro')
        print(pivot.to_markdown())
    
    # 2. Statistical Significance Testing
    if os.path.exists("ensemble_predictions.csv"):
        preds = pd.read_csv("ensemble_predictions.csv")
        y_true = preds['y_true'].values
        lr = preds['lr_preds'].values
        bert = preds['bert_preds'].values
        ensemble = preds['ensemble_preds'].values
        
        print("\n--- Statistical Significance (McNemar Test) ---")
        
        stat, p = mcnemar_test(y_true, lr, bert)
        print(f"Logistic vs BERT: stat={stat:.3f}, p-value={p:.4f}")
        
        stat, p = mcnemar_test(y_true, lr, ensemble)
        print(f"Logistic vs Ensemble: stat={stat:.3f}, p-value={p:.4f}")
        
        stat, p = mcnemar_test(y_true, bert, ensemble)
        print(f"BERT vs Ensemble: stat={stat:.3f}, p-value={p:.4f}")
        
        if p < 0.05:
            print("\nConclusion: The performance difference is STATISTICALLY SIGNIFICANT (p < 0.05).")
        else:
            print("\nConclusion: The performance difference is NOT statistically significant.")

    # Save summary to markdown
    with open("SCIENTIFIC_SUMMARY.md", "w") as f:
        f.write("# Scientific Evaluation Summary\n\n")
        f.write("## 1. Baseline Metrics (5-fold CV)\n")
        if os.path.exists("baseline_cv_results.csv"):
            f.write(avg_table.to_markdown(index=False))
        
        f.write("\n\n## 2. Statistical Significance\n")
        if os.path.exists("ensemble_predictions.csv"):
            f.write(f"- **Logistic vs Ensemble p-value**: {p:.4f}\n")
            f.write("- **Method**: McNemar's Test with continuity correction.\n")

if __name__ == "__main__":
    main()
