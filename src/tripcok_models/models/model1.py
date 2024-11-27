# model1.py

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
import nltk
import os

nltk.download('punkt')
DEFAULT_PATH='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'

def collect_from_batch(path):
    all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    data_frames = []
    for file in all_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens
    if word.isalpha() and word not in ENGLISH_STOP_WORDS]
    return tokens

def model_train():
    w2v_model = Word2Vec(
        sentences=df['tokens'], 
        vector_size=100,
        window=5,
        min_count=2,
        sg=0,
        workers=4
    )
    return w2v_model

def aggregate_vectors(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeroes(model.vector_size)
    
def recommend(destination, top_n=5):
    if destination not in df['tourist destination'].values:
        raise ValueError(f"Destination '{destination}' not found in dataset.")
    
    query_vector = df.loc[
        df['tourist destination'] == destination, 
        'vector'
        ].values[0].reshape(1, -1)

    vectors = np.vstack(df['vector'].values)
    similarities = cosine_similarity(query_vector, vectors)[0]

    df['similarity'] = similarities
    recommendations = df.sort_values('similarity', ascending=False).head(top_n + 1)  # +1 to exclude itself
    return recommendations[['tourist destination', 'similarity']].iloc[1:]  # Exclude the queried destination

def main():
    df = collect_from_batch()
    print(df.head)

main()