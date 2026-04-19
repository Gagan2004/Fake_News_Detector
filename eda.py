import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv("merged_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Label distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df)
    plt.title('Label Distribution (0: Real, 1: Fake)')
    plt.savefig('label_distribution.png')
    print("Saved label_distribution.png")
    
    # Text length analysis
    df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))
    print("\nText Length Stats:")
    print(df['text_len'].describe())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['text_len'] < 1000]['text_len'], bins=50)
    plt.title('Text Length Distribution (Truncated at 1000 words)')
    plt.savefig('text_length_distribution.png')
    print("Saved text_length_distribution.png")

if __name__ == "__main__":
    main()
