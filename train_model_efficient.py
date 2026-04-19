import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import numpy as np

def main():
    print("Loading merged dataset...")
    df = pd.read_csv("merged_dataset.csv")
    
    print("Pre-processing text...")
    df['text'] = df['text'].fillna('')
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    print("Vectorizing text (TF-IDF)... This might take a minute...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression model (with class balancing)...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    print("Evaluating...")
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Explainability: Feature Importance
    print("\n" + "="*30)
    print("TOP PREDICTORS OF FAKE NEWS")
    print("="*30)
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Sort features by coefficient magnitude
    # Positives are predictors for class 1 (Fake), Negatives for class 0 (Real)
    top_fake_indices = np.argsort(coefficients)[-20:][::-1]
    top_real_indices = np.argsort(coefficients)[:20]
    
    print("\nTop 20 words identifying FAKE news:")
    for idx in top_fake_indices:
        print(f" - {feature_names[idx]}: {coefficients[idx]:.4f}")
        
    print("\nTop 20 words identifying REAL news:")
    for idx in top_real_indices:
        print(f" - {feature_names[idx]}: {coefficients[idx]:.4f}")
    print("="*30 + "\n")
    
    print("Saving model and vectorizer...")
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, "models/fake_news_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("Done! Model saved in 'models/' directory.")

if __name__ == "__main__":
    main()
