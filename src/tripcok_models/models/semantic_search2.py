import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util
import pickle
import warnings

warnings.filterwarnings('ignore')
CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'


class CsvLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"\n[ERROR] CsvLoader failed to find any .csv files in {self.csv_path}.\n")
        
        self.df_list = sorted([
            entry.name for entry in os.scandir(csv_path)
            if entry.is_file() and entry.name.endswith('.csv')
        ])

    def fetch_df(self):
        data_frames = []
        for file in self.df_list:
            file_path = os.path.join(self.csv_path, file)
            try:
                df = pd.read_csv(file_path)[['contentid', 'title', 'overview']]
                data_frames.append(df)
            except FileNotFoundError:
                logging.warning(f"[ERROR] could not find file {file}. Skipping...")
                continue
        
        return pd.concat(data_frames, ignore_index=True)


class SemanticSearch:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.embedder = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.corpus_embeddings = None
        self.contentid_map = []

    def add_corpus(self, overviews: pd.Series, contentids: pd.Series):
        self.contentid_map = contentids.tolist()
        embeddings = []
        data_loader = DataLoader(overviews, batch_size=self.batch_size)

        for batch in data_loader:
            batch_embeddings = self.embedder.encode(batch, convert_to_tensor=True).cpu()
            embeddings.append(batch_embeddings)

        self.corpus_embeddings = torch.cat(embeddings, dim=0)

    def search(self, queries: list, top_k: int = 5, filter_list: list = None):
        if self.corpus_embeddings is None:
            raise ValueError("[ERROR] Corpus is empty. Add data using add_corpus().")

        all_results = []

        for query in queries:
            query_embedding = self.embedder.encode(query, convert_to_tensor=True).cpu()
            cos_scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()

            if filter_list:
                filter_indices = [
                    self.contentid_map.index(cid)
                    for cid in filter_list
                    if cid in self.contentid_map
                ]
            else:
                filter_indices = []

            query_index = (
                self.contentid_map.index(query) if query in self.contentid_map else None
            )

            top_k = min(
                len(self.contentid_map),
                top_k + len(filter_indices) + (1 if query_index is not None else 0)
            )
            top_results = torch.topk(cos_scores, k=top_k).indices.numpy()

            cids = []
            for idx in top_results:
                if idx == query_index or idx in filter_indices:
                    continue    
                cids.append(self.contentid_map[idx])
                if len(cids) >= top_k:
                    break
            
            all_results.append(cids)
        
        return all_results

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


def main():
    logging.basicConfig(level=logging.INFO)

    model_path = "semantic_search.pkl"
    csv_path = CSV_PATH
    model_name = "jhgan/ko-sroberta-multitask"
    searcher = None

    if os.path.exists(model_path):
        logging.info(f"[INFO] Loading existing model from {model_path}.")
        searcher = SemanticSearch.load_model(model_path)
    else:
        logging.info(f"[INFO] No existing model found. Creating new model.")
        csv_loader = CsvLoader(csv_path)
        df = csv_loader.fetch_df()

        searcher = SemanticSearch(model_name=model_name)
        searcher.add_corpus(df['overview'], df['contentid'])
        searcher.save_model(model_path)
    
    # Testing
    logging.info(f"[DEBUG] Testing with a random query...")
    random_row = df.sample(1).iloc[0]
    query = random_row['overview']
    logging.info(f"[DEBUG] Query: {query}")
    results = searcher.search([query], top_k=5)

    for idx, cid in enumerate(results[0], start=1):
        logging.info(f"[RESULT {idx}] Content ID: {cid}")


if __name__ == "__main__":
    main()
