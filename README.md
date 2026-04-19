# Fake News Detector 🕵️‍♂️

A multi-dataset fake news detection system that integrates diverse news sources into a unified classification model. This project leverages **ISOT**, **LIAR**, **FEVER**, and **FakeNewsNet** datasets to train a robust detector.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- All raw datasets placed in their respective folders (see [Project Structure](#-project-structure))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Gagan2004/Fake_News_Detector.git
   cd Fake_News_Detector
   ```

2. **Set up a virtual environment:**
   ```bash
   # Create venv
   python -m venv venv

   # Activate venv (Windows)
   .\venv\Scripts\activate

   # Activate venv (Mac/Linux)
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage Guide

Follow these steps in order to process the data and train the model:

### Step 1: Data Unification
Merge the four datasets into a single standardized CSV file.

##IMPORTANT --

  LINK FOR THE DATASETS ;
  https://drive.google.com/drive/folders/1Vn-lvU80o0Q3CioFA3VIcY2Dol7ClWNR?usp=sharing

  put each of the dataset folder in the root directory before running any script ;
  
```bash
python data_unification.py
```
*This will generate `merged_dataset.csv` with 167k+ records.*

### Step 2: Exploratory Data Analysis (Optional)
Generate distribution plots and statistical insights.
```bash
python eda.py
```

### Step 3: Training the Model
You have two options depending on your hardware:

**Option A: Efficient Training (Recommended for CPU)**
Trains a Logistic Regression model with TF-IDF vectorization. Fast and highly effective (81% accuracy).
```bash
python train_model_efficient.py
```

**Option B: Deep Learning (Requires GPU)**
Fine-tunes a DistilRoBERTa model. 
```bash
python train_model_manual.py
```

### Step 4: Evaluation
Test the final model on manual news headlines or unseen data.
```bash
python evaluate_model_efficient.py
```

---

## 📂 Project Structure

- `data_unification.py`: Merges raw datasets into one.
- `train_model_efficient.py`: Optimized training for all-cpu environments.
- `evaluate_model_efficient.py`: Scoring and manual testing script.
- `PROJECT_ARCHITECTURE.md`: Detailed technical documentation.
- `models/`: Directory containing serialized model artifacts.
- `ISOT_News_Dataset/`, `LIAR/`, `FEVERDataset/`, `FakeNewsNet/`: (Ignored folders) Raw data storage.

---

## 📊 Performance
The system achieves a consistent **81%+ accuracy** across a diverse test set encompassing both short claims and long-form articles. It includes built-in **explainability** to show which words (like "Reuters" or "Watch") drive the predictions.

## 📄 License
See the `LICENSE.md` file for details (if applicable).
