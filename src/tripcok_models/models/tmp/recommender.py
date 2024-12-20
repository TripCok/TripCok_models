# recommender.py

#import torch
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

EMBEDDINGS = "word_embeddings.npy"
DESTINATIONS = "/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/saved_5.csv"
DEFAULT_PATH='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'

test_input = {
    'keywords': '스키 호텔 산 휴양 테마공원 강원도',
}

def collect_from_batch(path=DEFAULT_PATH):
    all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    data_frames = []
    for file in all_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)

    collected = pd.concat(data_frames, ignore_index=True)
    return collected

def recommend_destination(user_input=test_input, k=5):
    keywords = user_input['keywords'].split()      # list of keywords

    user_embedding = np.mean([
        embedding_matrix[np.random.randint(len(embedding_matrix))]
        for _ in keywords
    ], axis=0)

    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(embedding_matrix)

    distances, indices = knn.kneighbors([user_embedding])


    print(f"Indices of recommended destinations: {indices}")
    print(f"Distances of recommended destinations: {distances}")

    destination_data_reset = destination_data.reset_index(drop=True)

    if len(indices[0]) > 0:
        recommendations = destination_data.iloc[indices[0]]
    else:
        print("No valid recommendations found.")
        recommendations = pd.DataFrame()  # Empty dataframe if no valid indices

    return recommendations [['title', 'overview']]

embedding_matrix = np.load(EMBEDDINGS)
destination_data = collect_from_batch()

recommendations = recommend_destination()

print(f"test_input:")
print(test_input)
print()
print(f"Recommended destinations:")
print(recommendations)
