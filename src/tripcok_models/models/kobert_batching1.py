# kobert_batching1.py
import os
import pandas as pd
import torch
import numpy as np
from kobert_transformers import get_kobert_model, get_tokenizer
from threading import Thread, Event
# libraries for logging
import logging
import threading
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
        self.thread = threading.Thread(target=self.animate, daemon=True)
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
    def __init__(self, csv_path: str, keyword_path: str, working_batch_size: int=32):
        self.csv_path = csv_path
        self.keyword_path = keyword_path
        self.working_batch_size = working_batch_size

        if not os.path.exists(csv_path):
            raise FileNotFoundError (f"\n[ERROR] BatchLoader failed to find any .csv files.\n")
        
        # .csv 파일 이름 목록 -> 누락/소실된 번호 존재하기 때문
        self.df_list = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
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
    def __init__(self):
        pass



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
        logger.info("Testing...")

        df = batch_loader.fetch_current_df()
        while df is not None:
            batch_loader.update()
            df = batch_loader.fetch_current_df()

            #if batch_loader.current_idx % 100 == 0:
            #    print(df[['title', 'overview']].head())
    
        logging.info(f"\n[DEBUG] Found {batch_loader.current_idx} number of .csv files.")
    finally:
        handler.stop()


if __name__ == "__main__":
    main()