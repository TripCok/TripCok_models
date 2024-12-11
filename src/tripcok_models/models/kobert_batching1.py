# kobert_batching1.py
import os
import pandas as pd
import torch
import numpy as np
from kobert_transformers import get_kobert_model, get_tokenizer
from threading import Thread#, Event
import gc # garbage collection
# libraries for logging
import logging
import time
import sys

# 로딩 애니메이션
class LoadingAnimationHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.animation_chars = "|/-\\"
        self.current_idx = 0
        self.is_running = False
        self.thread = None
        self.last_message = ""
    
    def emit(self, record):
        if not self.is_running: 
            return
        self.last_message = record.getMessage()
    
    def start(self):
        self.is_running = True
        self.thread = Thread(target=self.animate, daemon=True)
        self.thread.start()
    
    def animate(self):
        while self.is_running:
            sys.stdout.write(f"\r{self.last_message} {self.animation_chars[self.current_idx]}")
            sys.stdout.flush()
            self.current_idx = (self.current_idx +1) % len(self.animation_chars)
            time.sleep(0.2)
    
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\nComplete!\n")
        sys.stdout.flush()

# 파일 read
class BatchLoader():
    def __init__(self, csv_path: str, keyword_path: str, working_batch_size: int=16):
        self.csv_path = csv_path
        self.keyword_path = keyword_path
        self.working_batch_size = working_batch_size

        if not os.path.exists(csv_path):
            raise FileNotFoundError (f"\n[ERROR] BatchLoader failed to find any .csv files.\n")
        
        # .csv 파일 이름 목록 -> 누락/소실된 번호 존재하기 때문
        self.df_list = sorted([
            entry.name for entry in os.scandir(csv_path)
            if entry.is_file() and entry.name.endswith('.csv')
        ])
        self.current_idx = 0

        if not os.path.exists(keyword_path):
            os.makedirs(keyword_path, exist_ok=True)
            logging.info(f"\n[BatchLoader] keyword destination path {keyword_path} created.\n")
            logging.info(f"\n[BatchLoader] new path: starting from batch 0.\n")
        else:
            logging.info(f"\n[BatchLoader] successfully found keyword destination path {keyword_path}.\n")
            discovered = len([
                f for f in os.listdir(keyword_path)
                if f.endswith('.parquet')
            ])
            # 0개 찾은 경우 -> batch_0 부터
            # n개 찾은 경우 -> batch_n : n+1 번째 batch 부터
            if discovered > 0:
                logging.info(f"\n[BatchLoader] discovered {discovered} number of completed .parquet files.\n")
                logging.info(f"\n[BatchLoader] updating current working batch to {discovered}.\n")
                self.current_idx = discovered
            else:
                logging.info(f"\n[BatchLoader] did not find any complete batches. starting from 0.\n")


    # returns False when reaches end of csv
    def fetch_current_df(self):
        if self.current_idx >= len(self.df_list):
            return None                 # End of csv
        try:
            df = pd.read_csv(os.path.join(self.csv_path, self.df_list[self.current_idx]))
            return df[[
                'contentid',
                'title',
                'overview',
                'cat3',
                'region'
            ]]
        except FileNotFoundError:
            logging.info(f"\n[ERROR] could not find file {self.df_list[self.current_idx]}. It really should exist.")
            return pd.dataFrame()       # check if return df is empty upon receiving
        
    def update(self):
        self.current_idx += 1

# 작업클래스
class KobertExtractor:
    def __init__(self, keyword_path: str, top_n: int=10, batch_size=8):
        self.keyword_path = keyword_path
        self.top_n = top_n
        self.batch_size = batch_size
        self.model = get_kobert_model()
        self.tokenizer = get_tokenizer()
        
        self.model.eval()
        #self.stop_event = Event()
        #self.animation_thread = None

    # for a given batch, extract keywords
    def process_batch(self, batch: pd.DataFrame):
        texts = [
            text for text in batch['overview'].tolist() 
            if isinstance(text, str) and len(text.strip()) > 0
        ]
        if not texts:
            logging.warning("Skipping batch with no valid text data.")
            return []
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Extract attention scores
        attention = outputs.attentions
        avg_attention = torch.stack(attention).mean(dim=0)

        # Extract keywords
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())
        cls_attention = avg_attention[0, :, 0].detach().cpu().numpy()

        if len(tokens) != len(cls_attention):
            logging.error(f"Token-Attention mismatch: {len(tokens)} tokens, {len(cls_attention)} attention scores.")
            logging.error(f"Tokens: {tokens}")
            logging.error(f"CLS attention: {cls_attention}")
            return []

        special_token_ids = self.tokenizer.all_special_ids
        token_attention_pairs = [
            (token, score)
            for token, score, token_id in zip(tokens, cls_attention, inputs['input_ids'][0].cpu().numpy())
            if token_id not in special_token_ids and not np.isnan(score) and score > 0.01
        ]
        sorted_pairs = sorted(token_attention_pairs, key=lambda x: x[1], reverse=True)
        top_keywords = [pair[0] for pair in sorted_pairs[:self.top_n]]

        return top_keywords

    def extract_keywords(self, df: pd.DataFrame):
        # Split the dataframe into smaller batches
        batches = [df.iloc[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
        all_keywords = []

        for batch in batches:
            keywords = self.process_batch(batch)
            all_keywords.extend(keywords)  # Add the results from this batch
            
            # Clear memory after processing the batch
            del batch, keywords

        return all_keywords

            


CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'
KEYWORD_PATH = '/home/nishtala/TripCok/TripCok_models/src/tripcok_models/models/parquet/keywords'


def main():
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    handler = LoadingAnimationHandler()
    logger.addHandler(handler)

    try:
        batch_loader = BatchLoader(csv_path=CSV_PATH, keyword_path=KEYWORD_PATH)       
        handler.start()
        logger.info("Processing...")

        #df = batch_loader.fetch_current_df()
        #while df is not None:
        #    batch_loader.update()
        #    df = batch_loader.fetch_current_df()

            #if batch_loader.current_idx % 100 == 0:
            #    print(df[['title', 'overview']].head())
    
        logging.info(f"\n[DEBUG] Found {batch_loader.current_idx} number of .csv files.")
    finally:
        handler.stop()

def main_single_batch():
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    handler = LoadingAnimationHandler()
    logger.addHandler(handler)

    try:
        batch_loader = BatchLoader(csv_path=CSV_PATH, keyword_path=KEYWORD_PATH)
        handler.start()
        logger.info("Processing a single batch...")

        df = batch_loader.fetch_current_df()
        if df is not None:
            kobert_extractor = KobertExtractor(keyword_path=KEYWORD_PATH)

            logger.info("[DEBUG] Extracting keywords from the first batch...")
            single_batch_keywords = kobert_extractor.process_batch(df.iloc[:kobert_extractor.batch_size])
            
            logger.info(f"[DEBUG] Keywords for the first batch:\n{single_batch_keywords}")
        else:
            logger.error("No data found in the first batch.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        handler.stop()

def test_single_batch():
    # Initialize paths and batch loader
    batch_loader = BatchLoader(csv_path=CSV_PATH, keyword_path=KEYWORD_PATH)
    kobert_extractor = KobertExtractor(keyword_path=KEYWORD_PATH, top_n=5)

    # Fetch a single batch
    single_batch_df = batch_loader.fetch_current_df()
    if single_batch_df is not None:
        logging.info("[DEBUG] Extracting keywords from the first batch...")
        try:
            keywords = kobert_extractor.process_batch(single_batch_df.head(1))  # Process a single row for simplicity
            print(f"Extracted keywords: {keywords}")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No data to process!")



if __name__ == "__main__":
#    main()
#    main_single_batch()
    test_single_batch()