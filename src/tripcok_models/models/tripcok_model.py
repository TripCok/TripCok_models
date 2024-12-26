import os
import logging
import torch
from sentence_transformers import util
import pickle
import warnings

warnings.filterwarnings('ignore')

class SemanticSearch:
    def __init__(self, batch_size=128):
        self.embedder = None
        self.batch_size = batch_size
        self.corpus_embeddings = None
        self.contentid_map = []

    def search_cid(self, contentids: list, top_k: int=5, similarity_threshold: float=0.5):
        """
        Takes a list of contentids
        returns a list of dictionaries in format {cid: score}
        """
        if self.corpus_embeddings is None:
            raise ValueError("[ERROR] Empty corpus: Model Uninitialized!")

        result_dict_list = []
        input_cid_indices = [
            idx for idx, cid in enumerate(self.contentid_map) 
            if cid in contentids
        ]

        #logging.info(f"[DEBUG] search_cid: found cid indices {input_cid_indices}\n")
        if not input_cid_indices:
            logging.warning("[WARNING] No matching contentids found.")
            return result_dict_list
        
        for input_index in input_cid_indices:
            result_dict = {}
            embedding = self.corpus_embeddings[input_index]
            
            cos_scores = util.pytorch_cos_sim(embedding, self.corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()
            top_results = torch.topk(cos_scores, k=top_k+1).indices.numpy()

            for result_index in top_results:
                if result_index == input_index: continue

                result_cid = self.contentid_map[result_index]
                result_dict[result_cid] = cos_scores[result_index].item()

                if len(result_dict) >= top_k:
                    break
            
            result_dict_list.append(result_dict)
    
        return result_dict_list

    @staticmethod
    def load_model(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

def main():
    logging.basicConfig(level=logging.INFO)

    # --- Lazy Import for CsvLoader & pandas
    from semantic2 import CsvLoader
    import pandas as pd
    csv_path = "/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/"
    csv_loader = CsvLoader(csv_path)
    df = csv_loader.fetch_df()

    model_path = "semantic_search.pkl"    
    searcher = None
    
    if os.path.exists(model_path):
        logging.info(f"[INFO] Loading existing model from {model_path}.")
        searcher = SemanticSearch.load_model(model_path)
    else:
        logging.warning(f"[WARNING] Invalid model path. Exiting...")
        return
    
    print(df.dtypes)

    # --- Testing
    #"""
    query_num = 3
    print(f"[DEBUG] Testing model for {query_num} queries, top_k: {5}...\n")
    
    input_cid_list = []
    for i in range(query_num):
        random_row = df.sample(1).iloc[0]
        input_cid_list.append(random_row['contentid'])
    
    # results
    result_dict_list = searcher.search_cid(input_cid_list)

    # --- Test Print
    for i, (cid, result_dict) in enumerate(zip(input_cid_list, result_dict_list)):
        print(f"[INFO] Input #{i}: {cid}: {df.loc[df['contentid'] == cid, 'title'].iloc[0]}")
        print(f"[INFO] Input text: {df.loc[df['contentid'] == cid, 'overview'].iloc[0].partition('.')[0]}\n")
        for j, result_cid in enumerate(result_dict):
            result_title = df.loc[df['contentid'] == result_cid, 'title'].iloc[0]
            result_text = df.loc[df['contentid'] == result_cid, 'overview'].iloc[0]
            result_text = result_text.partition('.')[0]
            score = result_dict[result_cid]

            print(f"[DEBUG] RESULT #{j}: {result_cid}: {result_title}")
            print(f"[DEBUG] Similarity Score: {score}")
            print(f"[DEBUG] Text: {result_text}...\n")
        print("*"*50)
        print()
    #"""


if __name__ == "__main__":
    main()