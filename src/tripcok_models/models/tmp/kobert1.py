# kobert1.py

import torch
import torch.nn as nn
import pandas as pd
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from kobert_transformers import get_kobert_model, get_tokenizer
from transformers import BertTokenizer
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

import sys
import time

import threading

DEFAULT_PATH='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'

stop_words_kr = set([
    '있다', '있고', '있으며', '하는', '하다', '아니라', '이다', '이',
    '그', '그것', '또한', '같은', '따라', '때문에', '와', '이어서',
    '로', '을', '를', '이', '가', '의', '에', '에서', '와', '과', '에',
    '가', '인', '부터', '까지', '부터', '로서', '이나', '거나', '다른',
    '이곳은', '곳이다', '있도록', '이용할', '있어', '많이', '위한', '위해',
    '하는', '하여', '한다', '만든', '매년', '통해', '바로', '아니라',
    '좋다', '않도록', '아니라', '외에도', '중심으로', '각종', '여러', '가장',
    '가능한', '갖추고', '것으로', 'br', '<br>', '되었다', '당시', '대한',
    '있는', '이후', '현재', '함께', '등이', '가능하다', '것을', '것이',
    '인근에', '거리에', '것이다', '된다', '주변', '마련되어', '가는', '곳으로',
    '공간이다', '등의', '넓은', '모두', '등을', '즐길', '남아', '모든', '그리고',
    '곳으로', '공간이다', '규모의', '가지', '감상할', '건물'
])
patterns = re.compile(r"""
    습니다|다|요|어요|죠|네|겠|네요|니까|고|을|를|은|
    는|이|가|에|으로|에서|와|과|야|든|의|했|으며|으면|등도|들도|
    으며|었던|였|었|cm|ow|으나|하게|
    해서|하지|하면|되어|[0-9]호선|되면서
    """, re.VERBOSE)

tokenizer = get_tokenizer()
model = get_kobert_model()
model.eval()

def tokenize_korean(text):
    tokens = tokenizer.tokenize(text)
    return tokens


def clean_text(text):
    global stop_words_kr
    global patterns

    tokens = tokenize_korean(text)
    filtered_tokens = [
        re.sub(patterns, "", word)
        for word in tokens
        if word not in stop_words_kr and word.isalpha()
    ]
    return ' '.join(filtered_tokens)


def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.squeeze(0)

    return embeddings, inputs['input_ids']


def extract_keywords(text, top_n=100):
    embeddings, input_ids = get_embeddings(text)
    embeddings = embeddings.numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    avg_embedding = np.mean(embeddings, axis=0) # average for context
    cosine_similarities = cosine_similarity(embeddings, avg_embedding.reshape(1, -1)).flatten()

    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    keywords = [
        tokens[idx] for idx in top_indices
        if tokens[idx] != '[PAD]' and tokens[idx] not in stop_words_kr
    ]
    return keywords


def clean_tokenized_text(tokens):
    return [token.replace('_', '') for token in tokens]


def collect_from_batch(path=DEFAULT_PATH):
    all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"[ERROR] no .csv files found in: {path}")
    
    data_frames = []
    for file in all_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    
    collected = pd.concat(data_frames, ignore_index=True)
    collected = collected[['title', 'overview', 'cat3', 'region']]
    return collected


def apply_tf_idf(df, top_n_words=10):
    translator = str.maketrans('', '', string.punctuation)
    
    df['overview'] = df['overview'].apply(
        lambda x: x.translate(translator)
        if isinstance(x, str) else ''
    )
    df['overview'] = df['overview'].apply(
        lambda x: clean_text(str(x))
    )

    tf_idf_vectorizer = TfidfVectorizer(max_features=top_n_words)
    tfidf_matrix = tf_idf_vectorizer.fit_transform(df['overview'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns = tf_idf_vectorizer.get_feature_names_out())

    df['overview'] = tfidf_df.apply(
        lambda row: row.nlargest(top_n_words).index.tolist(), axis=1
    )
    return df


def merge(df):
    df['overview'] = df.apply(
        lambda row: row['overview'] + [row['cat3'], row['region']], axis=1
    )
    df['overview'] = df['overview'].apply(
        lambda x: [word for word in x if word.lower() != 'unknown']
    )
    return df.drop(columns=['cat3', 'region'])


def preprocess_text(df, top_n_words=10):
    df = apply_tf_idf(df, top_n_words)
    return df


def extract_keywords_from_df(df, top_n=10):
    total_records = len(df)

    def keyword_extraction_with_progress(text, idx):
        if idx % 100 == 0 and idx > 0:
            print(f"\n[DEBUG] Processed {idx} records out of {total_records}.")
        return extract_keywords(' '.join(text), top_n=top_n)

    df['keywords'] = [
        keyword_extraction_with_progress(text, idx)
        for idx, text in enumerate(df['overview'])
    ]
    df['keywords'] = df['keywords'].apply(lambda x: clean_tokenized_text(x))

    df['keywords'] = df.apply(
        lambda row: row['keywords'] + [row['cat3'], row['region']], axis=1
    )


    return df


def loading_animation(message, stop_event):
    symbols = ["\\", "|", "/", "-"]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {symbols[idx]}")     # Overwrite the line
        sys.stdout.flush()
        idx = (idx + 1) % len(symbols)                      # Cycle through symbols
        time.sleep(0.2)                                     # animation speed
    sys.stdout.write("\r" + " "*(len(message) + 2) + "\r")  # Clear the line


def main():
    stop_event = threading.Event()

    animation_thread = threading.Thread(target=loading_animation, args=("Processing...", stop_event))
    animation_thread.start()

    print("\n[DEBUG] Starting data collection.")
    df = collect_from_batch()
    print(f"\n[DEBUG] Data collection completed: {len(df)} records fetched.")
    
    print("\n[DEBUG] Starting text preprocessing.")    
    df = preprocess_text(df, 100)
    print(f"\n[DEBUG] Text preprocessing completed. {len(df)} records processed.")

    print("\n[DEBUG] Starting keyword extraction.")
    df = extract_keywords_from_df(df, top_n=50)
    print(f"\n[DEBUG] Keyword extraction completed. {len(df)} keywords extracted.")


    stop_event.set()
    animation_thread.join()

    save_to = "preprocessed_keywords_1.csv"

    print(df.head())
    df.to_csv(save_to, index=False)
    print(f"\n[INFO] Results saved to {save_to}")

    print()
    print(f"\n[DEBUG] Saving results to {save_to}.")


main()