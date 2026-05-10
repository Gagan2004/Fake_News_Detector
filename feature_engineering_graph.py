import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import re

def extract_graph_features(text, top_keywords, window_size=5):
    if not isinstance(text, str) or not text.strip():
        return {
            'nodes': 0, 'edges': 0, 'avg_degree': 0, 
            'density': 0, 'components': 0, 'transitivity': 0
        }
    
    words = re.findall(r'\w+', text.lower())
    # Filter words to only include those in top_keywords for this document
    relevant_words = [w for w in words if w in top_keywords]
    
    if not relevant_words:
        return {
            'nodes': 0, 'edges': 0, 'avg_degree': 0, 
            'density': 0, 'components': 0, 'transitivity': 0
        }

    G = nx.Graph()
    G.add_nodes_from(set(relevant_words))
    
    # Add edges based on co-occurrence window
    for i in range(len(words)):
        if words[i] in top_keywords:
            for j in range(i + 1, min(i + window_size, len(words))):
                if words[j] in top_keywords:
                    G.add_edge(words[i], words[j])
    
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
        'density': nx.density(G),
        'components': nx.number_connected_components(G),
        'transitivity': nx.transitivity(G)
    }

def main():
    print("Loading data for graph feature extraction...")
    df = pd.read_csv("merged_dataset.csv")
    
    # For speed in this demo/script, we'll process a subset
    SUBSET = 5000 
    print(f"Processing subset of {SUBSET} rows...")
    df_subset = df.sample(SUBSET, random_state=42).reset_index(drop=True)
    
    print("Calculating TF-IDF to identify keywords...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_subset['text'].fillna(''))
    feature_names = vectorizer.get_feature_names_out()
    
    graph_results = []
    print("Extracting graph features...")
    for i in tqdm(range(len(df_subset))):
        # Get top 20 keywords for this document
        row = tfidf_matrix.getrow(i).toarray()[0]
        top_indices = np.argsort(row)[-20:]
        top_keywords = set([feature_names[idx] for idx in top_indices if row[idx] > 0])
        
        features = extract_graph_features(df_subset['text'].iloc[i], top_keywords)
        graph_results.append(features)
    
    graph_df = pd.DataFrame(graph_results)
    combined_df = pd.concat([df_subset, graph_df], axis=1)
    
    combined_df.to_csv("merged_with_graph_features.csv", index=False)
    print("\nGraph features extracted and saved to merged_with_graph_features.csv")
    print("\nAverage Graph Stats by Class:")
    print(combined_df.groupby('label')[['avg_degree', 'density', 'transitivity']].mean())

if __name__ == "__main__":
    main()
