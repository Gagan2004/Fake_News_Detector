import joblib
import os

MODEL_PATH = "models/fake_news_model.pkl"
VEC_PATH = "models/vectorizer.pkl"

def predict(text, model, vectorizer):
    vec_text = vectorizer.transform([text])
    pred = model.predict(vec_text)[0]
    probs = model.predict_proba(vec_text)[0]
    return pred, probs[pred]

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found. Please run train_model_efficient.py first.")
        return

    print("Loading efficient model...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)

    # (Text, Ground Truth) where 0=REAL, 1=FAKE
    test_examples = [
        ("The moon is made of green cheese and NASA has been hiding it for decades.", 1),
        ("Scientists at CERN discover a new particle that could explain dark matter.", 0),
        ("Unbelievable: Drinking coffee every morning makes you immortal!", 1),
        ("The Federal Reserve announced a 0.25% interest rate hike after their meeting.", 0),
        ("BREAKING: New study shows that eating chocolate every day reduces risk of all cancers by 99%.", 1),
        ("The United Nations holds an emergency session to discuss the escalating conflict in the Middle East.", 0),
        ("Alert! Secret government documents reveal that the internet will be shut down for 30 days starting tomorrow.", 1),
        ("A major tech company released a statement denying rumors of a massive data breach affecting millions.", 0),
        ("Watch: Local man claims he has proof that the Earth is actually shaped like a donut.", 1),
        ("The Supreme Court is expected to deliver a ruling on the high-profile environmental case by the end of the week.", 0),
        ("SHOCKING: Scientists find that human DNA is actually 10% alien according to leaked lab reports.", 1),
        ("International energy agency warns of a sharp increase in global oil prices due to production cuts.", 0),
        ("You won't believe it! A giant monster was spotted in the depths of the Pacific Ocean.", 1),
        ("The prime minister met with several world leaders to negotiate a new international trade agreement.", 0),
        ("New miracle drug derived from common weeds cures 50 different diseases instantly.", 1),
        ("A major airline announced the cancellation of all flights to several European cities due to adverse weather.", 0),
    ]

    print("\n" + "="*50)
    print("MANUAL EVALUATION SUMMARY")
    print("="*50)
    
    correct = 0
    total = len(test_examples)

    for ex, true_label in test_examples:
        pred, conf = predict(ex, model, vectorizer)
        label_pred = "FAKE" if pred == 1 else "REAL"
        label_true = "FAKE" if true_label == 1 else "REAL"
        
        is_correct = (pred == true_label)
        if is_correct:
            correct += 1
            status = "CORRECT"
        else:
            status = "WRONG"

        print(f"Text: {ex[:60]}...")
        print(f"Prediction: {label_pred} | True: {label_true} | Status: {status} (Conf: {conf:.2f})")
        print("-" * 50)

    accuracy = (correct / total) * 100
    print(f"\nFINAL MANUAL SCORE: {correct}/{total} ({accuracy:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
