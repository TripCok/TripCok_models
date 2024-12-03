# model3.py

import torch
import torch.nn as nn           # neural network
import torch.optim as optim     # optimizer
import numpy as np
import pandas as pd
from collections import Counter
#import nltk
#from nltk.tokenize import word_tokenize
import random
import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import re   # regular expressions
from konlpy.tag import Okt

DEFAULT_PATH='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'
#nltk.download('punkt')
tokenizer = Okt()

def tokenize_korean(text):
    return tokenizer.morphs(text)

# consider saving to file and loading from there
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

pattern = r"(습니다|다|요|어요|죠|네|겠|네요|니까|고|을|를|은|는|이|가|에|으로|에서|와|과|야|로|든|의)"

class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecSkipGram, self).__init__()

        # instance of class nn.Embedding
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # configurable
        self.vocab_size = vocab_size

    # forward pass
    def forward(self, target, context):
        # call of nn.Embedding instance acts like a function call
        # because nn has a '__call__' method
        target_vec = self.target_embedding(target)
        context_vec = self.context_embedding(context)

        # cosine similarity: target & embedding, requires transpose
        scores = torch.matmul(target_vec, self.context_embedding.weight.t())
        log_probs = nn.functional.log_softmax(scores, dim=-1)

        return log_probs #scores


# retrieves batch data and merges into one
def collect_from_batch(path=DEFAULT_PATH):
    all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    if not all_files:
        raise FileNotFoundError(f"[ERROR] No .csv files found in directory: {path}")

    data_frames = []
    for file in all_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    
    collected = pd.concat(data_frames, ignore_index=True)
    collected = collected[['title', 'overview', 'cat3', 'region']]

    return collected


# clean stopwords1
def clean_text(text):
    global pattern
    global stop_words_kr

    tokens = tokenize_korean(text)

    filtered_tokens = [
        re.sub(pattern, "", word) for word in tokens
        if word not in stop_words_kr and word.isalpha()
    ]
    return ' '.join(filtered_tokens)

# preprocessing step 1:
# frequency analysis with TF-IDF
def apply_tf_idf(df, top_n_words=10):
    # cleans up string
    translator = str.maketrans('', '', string.punctuation)


    df['overview'] = df['overview'].apply(
        lambda x: str(x).translate(translator) if isinstance(x, str) else ''
    )
    df['overview'] = df['overview'].apply(
        lambda x: clean_text(str(x))
    )

    tf_idf_vectorizer = TfidfVectorizer(max_features=top_n_words)

    # applies TF-IDF to 'overview' column
    tfidf_matrix = tf_idf_vectorizer.fit_transform(df['overview'])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tf_idf_vectorizer.get_feature_names_out()
    )
    df['overview'] = tfidf_df.apply(
        lambda row: row.nlargest(top_n_words).index.tolist(), axis=1
    )
    return df


# preprocessing step 2:
# merge columns
def merge(df):
    df['overview'] = df.apply(
        lambda row: row['overview'] + [row['cat3'], row['region']],
        axis = 1
    )

    df['overview'] = df['overview'].apply(
        lambda x: [word for word in x if word.lower() != 'unknown']
    )

    return df.drop(columns=['cat3', 'region'])


# the actual preprocessing func
def preprocess_text(df, top_n_words = 10):
    df = apply_tf_idf(df, top_n_words)
    df = merge(df)
    return df


def create_vocab(corpus):
    vocab, reverse_vocab = {}, {}
    idx = 0

    for sentence in corpus:
        for word in sentence:
            if word not in vocab:
                vocab[word] = idx
                reverse_vocab[idx] = word
                idx += 1
    return vocab, reverse_vocab


# maybe??
def train_word2vec(
    model, corpus, vocab, reverse_vocab, 
    embedding_dim=100, window_size=2, epochs=10,
    learning_rate=0.01, batch_size=128, debug_interval=1000
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLoss()
    
    dataset = []
    for sentence in corpus:
        indices = [vocab[word] for word in sentence if word in vocab]
        for i, target_idx in enumerate(indices):
            context_indices = [
                indices[j] for j in range(max(0, i-window_size), min(len(indices), i+window_size+1))
                if j != i
            ]
            dataset.append((target_idx, context_indices))

    print(f"[INFO] Dataset size: {len(dataset)}")

    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(dataset)
        for i, (target, context) in enumerate(dataset):
            target_tensor = torch.tensor([target], dtype=torch.long)
            context_tensor = torch.tensor(context, dtype=torch.long)

            optimizer.zero_grad()
            log_probs = model(target_tensor, context_tensor)
            loss = loss_fn(log_probs, context_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i% debug_interval == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")
        print(f"[INFO] Epoch {epoch+1}, Total Loss: {total_loss:.4f}")

#
def search_embedding():
    pass



#
def main():
    df = collect_from_batch()
    df = preprocess_text(df, 100)

    # debug checkpoint 1
    #print(df.head())
    #print("...")
    #print(df.tail())
    #df.to_csv(os.path.join(os.getcwd(), "preprocessed1.csv"), index=False)
    #print()
    #print(f"[INFO] saved to preprocessed1.csv")

    print("[DEBUG] first few processed overviews:")
    print(df['overview'].head())

    print(f"[INFO] Preprocessing complete.")

    corpus = df['overview']
    vocab, reverse_vocab = create_vocab(corpus)

main()