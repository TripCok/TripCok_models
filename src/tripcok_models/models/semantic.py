import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util
import pickle
import warnings

import torch.nn.functional as F

warnings.filterwarnings('ignore')

CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'
EMBEDDINGS_PATH = "corpus_embeddings.pt"
CONTENTID_PATH = "contentid_map.pkl"

class CsvLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"\n[ERROR] CsvLoader failed to find path {self.csv_path}.\n")
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
                # data clean
                df = df.dropna(subset=['overview'])
                df = df[df['overview'].str.strip() != ""]
                data_frames.append(df)
            except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                logging.warning(f"[ERROR] could not process file {file}: {str(e)}. Skipping...")
                continue

        return pd.concat(data_frames, ignore_index=True)

class SemanticSearch:
    def __init__(self, model_name: str, batch_size=128):
        self.embedder = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.corpus_embeddings = None
        self.contentid_map = []

    def add_corpus(self, overviews: pd.Series, contentids: pd.Series):
        if overviews.empty:
            raise ValueError("[ERROR] Overview data is empty. Cannot proceed with embedding.")
        self.contentid_map = contentids.tolist()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = self.embedder.to(device)

        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_info_path = os.path.join(checkpoint_dir, "checkpoint_info.pkl")
        start_batch = 0
        all_embeddings = []

        if os.path.exists(checkpoint_info_path):
            logging.info("[INFO] Found checkpoint. Attempting recovery...")
            with open(checkpoint_info_path, 'rb') as f:
                checkpoint_info = pickle.load(f)
                start_batch = checkpoint_info.get("last_batch", 0)
                checkpoint_files = checkpoint_info.get("saved_files", [])
            
            for file in checkpoint_files:
                try:
                    embeddings = torch.load(file)
                    all_embeddings.append(embeddings)
                except Exception as e:
                    logging.warning(f"[WARNING] Skipping  corrputed checkpoint file {file}: {str(e)}")

            logging.info(f"[INFO] Recovered embeddings from {len(checkpoint_files)} files. Resuming batch at {start_batch}.")

        data_loader = DataLoader(overviews, batch_size=self.batch_size, num_workers = 4)

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i < start_batch:
                    continue
                try:
                    batch_embeddings = self.embedder.encode(
                        batch, convert_to_tensor=True, device=device
                    ).cpu()
                    all_embeddings.append(batch_embeddings)

                    checkpoint_file = os.path.join(checkpoint_dir, f"embeddings_checkpoint_{i}.pt")
                    torch.save(batch_embeddings, checkpoint_file)

                    checkpoint_info = {
                        "last_batch": i+1,
                        "saved_files": [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
                    }
                    with open(checkpoint_info_path, 'wb') as f:
                        pickle.dump(checkpoint_info, f)

                    if i%10 == 0:
                        logging.info(f"[INFO] Checkpoint saved for batch: {i}.")
                except Exception as e:
                    logging.error(f"[ERROR] Failed to process batch {i}: {str(e)}")
                    continue
        if all_embeddings:
            self.corpus_embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else None
        else:
            raise RuntimeError("[ERROR] No embeddings were recovered or generated.")

        if self.corpus_embeddings is None:
            raise RuntimeError("[ERROR] No embeddings were generated.")
        logging.info(f"[INFO] Successfully generated embeddings for {len(self.contentid_map)} items.")

    def search(self, queries: list, top_k: int=5, filter_list: list=None, similarity_threshold: float = 0.5):
        if self.corpus_embeddings is None:
            raise ValueError("[ERROR] Corpus is empty. Add data using add_corpus().")

        result_dict = {}
        for query in queries:
            # generate query embedding 
            query_embedding = self.embedder.encode(query, convert_to_tensor=True).cpu()
            cos_scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()

            # threshold -> softmax
            cos_scores[cos_scores < similarity_threshold] = 0
            #cos_scores = F.softmax(cos_scores, dim=0)

            filter_indices = []
            if filter_list:
                filter_indices = [
                    self.contentid_map.index(cid) for cid in filter_list
                    if cid in self.contentid_map
                ]

            # determine query index
            query_index = (
                self.contentid_map.index(query) 
                if query in self.contentid_map else None
            )

            # determine top-k
            top_k = min(
                len(self.contentid_map),
                top_k + len(filter_indices) + (1 if query_index is not None else 0)
            )

            # get results
            top_results = torch.topk(cos_scores, k=top_k).indices.numpy()

            for idx in top_results:
                if idx == query_index or idx in filter_indices:
                    continue    
                if cos_scores[idx] > 0:
                    result_dict[self.contentid_map[idx]] = cos_scores[idx].item()
                if len(result_dict) >= top_k:
                    break
        
    #    logging.info(f"TYPECHECK: {type(result_dict)}")
        return result_dict

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
    csv_loader = CsvLoader(csv_path)
    df = csv_loader.fetch_df()
    
    if os.path.exists(model_path):
        logging.info(f"[INFO] Loading existing model from {model_path}.")
        searcher = SemanticSearch.load_model(model_path)

    else:
        logging.info(f"[INFO] No existing model found. Creating new model.")

        # New SemanticSearch instance
        searcher = SemanticSearch(model_name=model_name)

        # Check and load preexisting embeddings
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(CONTENTID_PATH):
            logging.info("[INFO] Loading precomputed embeddings...")
            searcher.corpus_embeddings = torch.load(EMBEDDINGS_PATH)
            with open(CONTENTID_PATH, 'rb') as f:
                searcher.contentid_map = pickle.load(f)
                logging.info("[INFO] Loaded embeddings.\n")

        # add corpus to searcher
        searcher.add_corpus(df['overview'],df['contentid'])

        # save embeddings & content_map
        torch.save(searcher.corpus_embeddings, EMBEDDINGS_PATH)
        with open(CONTENTID_PATH, 'wb') as f:
            pickle.dump(searcher.contentid_map, f)

        logging.info(f"[INFO] Saving trained model to {model_path}.")
        searcher.save_model(model_path)

    # Testing
    logging.info(f"[DEBUG] Testing with a random query...")
    random_row = df.sample(1).iloc[0]
    query = random_row['overview']
    
    logging.info(f"[DEBUG] Query: {random_row['title']}")
    logging.info(f"[DEBUG] Query: {query}\n")

    results = searcher.search([query], top_k=5)

    for idx, cid in enumerate(results, start=1):
        title = df[df['contentid'] == cid]['title'].iloc[0]
        overview = df[df['contentid'] == cid]['overview'].iloc[0]
        score = results[cid]

        logging.info(f"[RESULT {idx}]")
        logging.info(f"Contentid: {cid}, similarity score: {score}")
        logging.info(f"Title: {title}\nOverview:{overview}\n")

if __name__ == "__main__":
    main()