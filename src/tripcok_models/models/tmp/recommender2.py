# recommender2.py

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

def main():
    destination_data = collect_from_batch()

    print(test_input)
    print()
    print(destination_data.head())
    print("...")
    print(destination_data.tail())


main()