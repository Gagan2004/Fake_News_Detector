import pandas as pd
import numpy as np
import os

def load_isot():
    print("Loading ISOT dataset...")
    true_df = pd.read_csv("ISOT_News_Dataset/True.csv")
    fake_df = pd.read_csv("ISOT_News_Dataset/Fake.csv")
    
    true_df['label'] = 0
    fake_df['label'] = 1
    
    df = pd.concat([true_df, fake_df])
    # Combine title and text
    df['text_combined'] = df['title'] + " " + df['text']
    return df[['text_combined', 'label']].rename(columns={'text_combined': 'text'})

def load_liar():
    print("Loading LIAR dataset...")
    xl = pd.ExcelFile("LIAR/liar_data.xlsx")
    dfs = []
    for sheet in xl.sheet_names:
        dfs.append(pd.read_excel(xl, sheet_name=sheet))
    df = pd.concat(dfs)
    
    # Mapping
    # True, mostly-true, half-true -> 0
    # False, barely-true, pants-fire -> 1
    # Note: LIAR labels in this file seem to be ['pants-fire' 'half-true' 'mostly-true' True False 'barely-true']
    def map_liar(label):
        label_str = str(label).lower().strip()
        if label_str in ['true', 'mostly-true', 'half-true']:
            return 0
        if label_str in ['false', 'barely-true', 'pants-fire']:
            return 1
        return None

    df['label'] = df['Label'].apply(map_liar)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    return df[['Text', 'label']].rename(columns={'Text': 'text'})

def load_fever():
    print("Loading FEVER dataset...")
    df = pd.read_csv("FEVERDataset/train.tsv", sep='\t')
    
    # SUPPORTS -> 0, REFUTES -> 1
    df = df[df['label'].isin(['SUPPORTS', 'REFUTES'])]
    df['label'] = df['label'].map({'SUPPORTS': 0, 'REFUTES': 1})
    return df[['claim', 'label']].rename(columns={'claim': 'text'})

def load_fakenewsnet():
    print("Loading FakeNewsNet dataset...")
    paths = [
        ("FakeNewsNet/BuzzFeed_real_news_content.csv", 0),
        ("FakeNewsNet/BuzzFeed_fake_news_content.csv", 1),
        ("FakeNewsNet/PolitiFact_real_news_content.csv", 0),
        ("FakeNewsNet/PolitiFact_fake_news_content.csv", 1)
    ]
    dfs = []
    for path, label in paths:
        if os.path.exists(path):
            temp_df = pd.read_csv(path)
            temp_df['label'] = label
            temp_df['text_combined'] = temp_df['title'].fillna('') + " " + temp_df['text'].fillna('')
            dfs.append(temp_df[['text_combined', 'label']])
    
    df = pd.concat(dfs)
    return df.rename(columns={'text_combined': 'text'})

def main():
    isot = load_isot()
    liar = load_liar()
    fever = load_fever()
    fnn = load_fakenewsnet()
    
    print(f"ISOT: {len(isot)}")
    print(f"LIAR: {len(liar)}")
    print(f"FEVER: {len(fever)}")
    print(f"FakeNewsNet: {len(fnn)}")
    
    merged_df = pd.concat([isot, liar, fever, fnn])
    print(f"Total merged: {len(merged_df)}")
    
    # Basic cleaning
    print("Cleaning text...")
    merged_df['text'] = merged_df['text'].fillna('').str.lower()
    
    # Save merged dataset
    merged_df.to_csv("merged_dataset.csv", index=False)
    print("Saved merged_dataset.csv")
    
    # Display distribution
    print("\nLabel Distribution:")
    print(merged_df['label'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
