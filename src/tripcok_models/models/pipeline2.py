# pipeline2.py
# kobert > pipeline > ???

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_PATH='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/models/'
VECTORIZE_BY = 'keywords' # 'overview' or 'keywords'
NEED_NEW_MODEL = True       # need to change main: if TRUE train new model else call

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_tfidf_vectors(df, column_name: str):
    if column_name not in {'overview', 'keywords'}:
        raise ValueError(f"Invalid column name: {column_name}. Must be one of 'overview' or 'keywords'.")

    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    ) if column_name == 'overview' else TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2)
    )

    if column_name == 'keywords':
        keyword_strings = df[column_name].apply(lambda x: ' '.join(eval(x)))  # Convert string lists back to text
        tfidf_matrix = tfidf_vectorizer.fit_transform(keyword_strings)
        return tfidf_matrix

    elif column_name == 'overview':
        overview_texts = df[column_name].fillna('')  # Replace NaN with empty strings
        tfidf_matrix = tfidf_vectorizer.fit_transform(overview_texts)
        return tfidf_matrix

    else:
        print("[ERROR] This message shouldn't be reachable!")
        return None

def train_knn(tfidf_matrix, n_neighbors=5):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(tfidf_matrix)
    return knn

def recommend(target_idx, knn_model, tfidf_matrix, data, n_recommendations=5):
 #   print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
 #   print(f"Sample TF-IDF Vector for Target Index {target_idx}:\n", tfidf_matrix[target_idx])
    
    distances, indices = knn_model.kneighbors(
        tfidf_matrix[target_idx].reshape(1, -1), n_neighbors=n_recommendations+1
    )
    filtered_indices  = [idx for idx in indices[0] if idx != target_idx][:n_recommendations]
    recs = data.iloc[filtered_indices]
    return recs[['title', 'region', 'cat3']]

def main():
    csv_name = "preprocessed_keywords_1.csv"
    data = load_data(os.path.join(DEFAULT_PATH, csv_name))

    print("Data loaded successfully. First few rows:")
    print(data[VECTORIZE_BY].head(10))
    
    # Create TF-IDF vectors using the 'overview' column
    print("\nCreating TF-IDF vectors for 'overview'...")
    tfidf_matrix = create_tfidf_vectors(data, VECTORIZE_BY)
#    print("TF-IDF matrix shape:", tfidf_matrix.shape)  # Verify TF-IDF dimensions
    
    # Train KNN model
    print("\nTraining KNN model...")
    knn_model = train_knn(tfidf_matrix, n_neighbors=5)
    print("KNN model trained.")
    
    target_idx = 0  # Choose a valid index in your dataset
    print(f"\nBeginning random test...")
    target_idx = random.randint(0, len(data)-1)
    print(f"\nTesting recommendation for target index {target_idx}:")

    print(f"{data.iloc[target_idx][['title', 'region', 'cat3']]}")
    try:
        recommendations = recommend(target_idx, knn_model, tfidf_matrix, data, n_recommendations=5)
        print("Recommendations:")
        print(recommendations)
    except Exception as e:
        print(f"[ERROR] Failed to generate recommendations: {e}")


main()