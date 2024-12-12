# model2.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import random
import os
import string

DEFAULT_PATH='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'
sample_data = []

nltk.download('punkt')

def collect_from_batch(path=DEFAULT_PATH):
    all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    data_frames = []
    for file in all_files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)

    collected = pd.concat(data_frames, ignore_index=True)
    collected = collected[['title', 'cat1', 'cat2', 'cat3', 'region', 'overview']]

    return collected # limit

#    return pd.concat(data_frames, ignore_index=True)

def preprocess_text(corpus):
    translator = str.maketrans('', '', string.punctuation)

    corpus = [sentence.split('.')[0] for sentence in corpus]
    cleaned_corpus = [sentence.translate(translator) for sentence in corpus]
    tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in cleaned_corpus]

    # flatten list of tokenized sentences / count word frequencies
    all_words = [word for sentence in tokenized_corpus for word in sentence]
    word_counts = Counter(all_words)

    # vocab: word -> index / reverse_vocab: index -> word
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    reverse_vocab = {idx: word for word, idx in vocab.items()}

    return tokenized_corpus, vocab, reverse_vocab

class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecSkipGram, self).__init__()
        
        # define embedding layers
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.vocab_size = vocab_size

    def forward(self, target, context):
        target_vec = self.target_embedding(target)
        context_vec = self.context_embedding(context)

        # cosine similarity (dot product -> requires transpose)
        scores = torch.matmul(target_vec, self.context_embedding.weight.t())
        log_probs = nn.functional.log_softmax(scores, dim=-1)
    #    return scores    
        return log_probs


def train_word2vec(
    model, corpus, vocab, reverse_vocab, 
    embedding_dim=100, window_size=2, epochs=10, 
    learning_rate=0.01, batch_size=128, debug_interval=1000
):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    training_data = []
    vocab_size = len(vocab)

    for sentence in corpus:
        # Generate training pairs (target, context)
        for idx, word in enumerate(sentence):
            target_word = word
            context_words = [sentence[i] for i in range(max(0, idx - window_size), min(len(sentence), idx + window_size + 1)) if i != idx]
            
            target_idx = vocab[target_word]
            context_indices = [vocab[context_word] for context_word in context_words]
            
            for context_idx in context_indices:
                training_data.append((target_idx, context_idx))
    
    # Step 3.2: Training loop
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data)
        
        num_prints=0
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            if len(batch) < batch_size:
                continue

            target_idx_batch = torch.tensor([target_idx for target_idx, _ in batch], dtype=torch.long)
            context_idx_batch = torch.tensor([context_idx for _, context_idx in batch], dtype=torch.long)

            optimizer.zero_grad()
            output = model(target_idx_batch, context_idx_batch)
            
            # Compute loss
            loss = loss_function(output, context_idx_batch)
            loss.backward()

            if (i+1) % debug_interval == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        print(f"[DEBUG] Gradient norm for {param.shape}: {param.grad.norm()}")

            optimizer.step()

            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.5)
            total_loss += loss.item()

            if i % debug_interval == 0:
                num_prints += 1
                print(f"[DEBUG] Epoch {epoch+1}/{epochs}, checkpoint {num_prints}, Loss: {loss.item()}") #Total loss: {total_loss}")
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')

def search_embedding(word, vocab, word_embeddings):
    if word in vocab:
        word_idx = vocab[word]
        print(f"[INFO] Embedding for '{word}': {word_embeddings[word_idx]}")
    else:
        print(f"[INFO] Word '{word}' not in vocabulary")

# Step 4: Initialize and train the model
def main():
    df = collect_from_batch()

    tokenized_corpus, vocab, reverse_vocab = preprocess_text(df['overview'].dropna())
    vocab_size = len(vocab)
    print(f"[DEBUG] Sample tokenized sentence from overview:\n{tokenized_corpus[0]}")
    print(f"[DEBUG] Vocabulary size: {vocab_size}")

    embedding_dim = 300         # configurable
    model = Word2VecSkipGram(vocab_size, embedding_dim)

    train_word2vec(
        model=model, 
        corpus=tokenized_corpus, 
        vocab=vocab, 
        reverse_vocab=reverse_vocab, 
        embedding_dim=embedding_dim,
        window_size=5,
        epochs=50,
        learning_rate=0.2,
        debug_interval=100,
    )

    # Step 5: Get word embeddings
    word_embeddings = model.target_embedding.weight.data.numpy()
    np.save("word_embeddings.npy", word_embeddings)     # mid-save

    search_embedding("해변", vocab, word_embeddings)

    # Example: Get the embedding for the word "word2vec"
#    word_idx = vocab["word2vec"]
#    embedding_for_word2vec = word_embeddings[word_idx]
#    print(f"Embedding for 'word2vec': {embedding_for_word2vec}")

def tester_main():
    df = collect_from_batch()
    print(df.head())
    print(df.tail())
    print(df.size)

main()
#tester_main()