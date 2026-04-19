# Fake News Detection Project Plan

## Phase 1: Data Preparation
1. Load datasets (ISOT, LIAR, FakeNewsNet)
2. Standardize columns:
   - text
   - label
3. Merge datasets
4. Clean text:
   - lowercase
   - remove URLs, punctuation
   - remove stopwords (optional)

---

## Phase 2: Exploratory Data Analysis (EDA)
1. Check class distribution
2. Analyze text length
3. Visualize word frequencies

** You’re looking for bias traps.if fake = 90%, your model will just learn to shout “FAKE” at everything.

---

## Phase 3: Preprocessing for Model
1. Split into train/val/test (70/15/15)
2. Tokenize using RoBERTa tokenizer
3. Pad/truncate sequences (max_len = 512)

---

## Phase 4: Model Building
1. Load pre-trained RoBERTa
2. Add classification head:
   - Dense → ReLU → Dropout → Output
3. Loss:
   - Binary Cross Entropy / Focal Loss

---

## Phase 5: Training
1. Optimizer: AdamW
2. Learning rate: 2e-5
3. Batch size: 16
4. Epochs: 3–5
5. Track:
   - loss
   - accuracy
   - F1-score

---

## Phase 6: Evaluation
1. Evaluate on test set
2. Metrics:
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
3. Confusion Matrix

---

## Phase 7: Advanced Enhancements
1. Add metadata features
2. Add social features
3. Use attention-based fusion

---

## Phase 8: Explainability
1. Use SHAP or LIME
2. Highlight important words

---

## Phase 9: Research Paper
1. Define problem statement
2. Describe methodology
3. Show experiments
4. Compare results
5. Highlight novelty

---

## Output
- Trained model
- Evaluation metrics
- Research paper draft
