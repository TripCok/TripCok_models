from kobert_transformers import get_kobert_model, get_tokenizer
import torch
import pandas as pd
import numpy as np
import os
#from sklearn.metrics.pairwise import cosine_similarity

# 로딩 애니메이션을 위한 라이브러리
import sys
import time
import threading

# batch 파일들의 저장 경로 - import 용
CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'
PQ_PATH = '/home/nishtala/TripCok/TripCok_models/src/tripcok_models/models/parquet/'

# KoBERT model and tokenizer
model = get_kobert_model(attn_implementation="eager")
tokenizer = get_tokenizer()
model.eval()


# .parquet 혹은 .csv에서 필요한 모든 파일을 하나의 데이터프레임으로 리턴
# 전체 엔트리가 51000여개로 많지 않으므로 용량 문제는 크게 없음
def collect_from_batch(pq_exists=False):
    # main 에서 경로체크 하므로 체크 필요 x
    path, extentions = (PQ_PATH, '.parquet') if pq_exists else (CSV_PATH, '.csv')

    all_files = [f for f in os.listdir(path) if f.endswith(extentions)]
    if not all_files:
        raise FileNotFoundError(f"[ERROR] no {extentions} files found in: {path}")
    
    data_frames = []
    for file in all_files:
        file_path = os.path.join(path, file)
        df = pd.read_parquet(file_path) if pq_exists else pd.read_csv(file_path)
        data_frames.append(df)
    
    collected = pd.concat(data_frames, ignore_index=True)
    collected = collected[[
        'contentid',            # Primary Key
        'title',                # 여행지 이름
        'overview',             # 개요 텍스트 -> 모델의 주 처리 대상
        'cat3',                 # 카테고리 소분류
        'region'                # 지역
    ]]
    return collected


# Function to tokenize and get attention scores
def extract_attention_scores(df, top_n=5, batch_size=8):
    keywords_list = []
    
    batch_size_actual = len(df)  # Set actual batch size from DataFrame
    for start_idx in range(0, batch_size_actual, batch_size):
        end_idx = min(start_idx + batch_size, batch_size_actual)  # Handle last batch
        batch_texts = df.iloc[start_idx:end_idx]['overview'].tolist()  # Batch of texts

        # Tokenize the batch
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs, output_attentions=True)
        
        # Extract attention scores (from all layers)
        attention = outputs.attentions
        avg_attention = torch.stack(attention).mean(dim=0)  # Average over layers
        
        # Focus on CLS token's attention to all tokens
        cls_attention = avg_attention[0, :, 0].cpu().numpy()  # CLS token attention (first token in sequence)

        # Ensure the batch size aligns with cls_attention
        cls_attention = cls_attention[:batch_size_actual]  # Slice to match actual batch size

        batch_keywords = []
        for idx in range(end_idx - start_idx):  # Loop through each entry in the batch
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][idx].cpu().numpy())
            cls_attention_text = cls_attention[idx]

            # Filter out special tokens
            special_token_ids = tokenizer.all_special_ids
            token_attention_pairs = [
                (token, score)
                for token, score, token_id in zip(tokens, cls_attention_text, inputs['input_ids'][idx].cpu().numpy())
                if token_id not in special_token_ids
            ]

            # Sort by attention score
            sorted_pairs = sorted(token_attention_pairs, key=lambda x: x[1], reverse=True)

            # Extract top_n keywords
            top_keywords = [pair[0] for pair in sorted_pairs[:top_n]]
            batch_keywords.append(top_keywords)

        keywords_list.extend(batch_keywords)

    df['keywords'] = keywords_list
    return df


# 로딩용 애니메이션 > 이거 작동은 하는중? 에 대한 체크
def loading_animation(message, stop_event):
    symbols = ["\\", "|", "/", "-"]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {symbols[idx]}")     # Overwrite the line
        sys.stdout.flush()
        idx = (idx + 1) % len(symbols)                      # Cycle through symbols
        time.sleep(0.2)                                     # animation speed
    sys.stdout.write("\r" + " "*(len(message) + 2) + "\r")  # Clear the line


# 파티션 .parquet로 저장하는 함수 > TF-IDF 때 한번 (중간저장), keyword 한번 (최종저장)
def save_partitioned_pq(df: pd.DataFrame, output_path: str, rows_per_partition: int):
    num_partitions = len(df) // rows_per_partition + (1 if len(df) % rows_per_partition else 0)

    for i in range(num_partitions):
        start_idx = i * rows_per_partition
        end_idx = min((i+1)* rows_per_partition, len(df)) 

        partition_df = df.iloc[start_idx:end_idx]
        file_name = f"partition_{i+1}.parquet"
        partition_df.to_parquet(os.path.join(output_path, file_name))

    print(f"\n[INFO] .parquet files partitioned to {output_path}")


# Sample Usage
def main():

    print("[DEBUG] Main function has started.\n")
    print("[DEBUG] Starting data collection.")

    keyword_path = PQ_PATH + "/keywords/"

    # 경로 체크
    if not os.path.exists(keyword_path):
        os.makedirs(keyword_path, exist_ok=True)
        print(f"\n[INFO] path {keyword_path} created.")
    else:
        print(f"\n[INFO] path {keyword_path} found.")

    # Extract keywords using attention scores
    print("[INFO] Starting attention-based extraction.")

    # 애니메이션 시작
    stop_event = threading.Event()
    animation_thread = threading.Thread(target=loading_animation, args=("Processing...", stop_event))
    animation_thread.start()

    if not os.listdir(keyword_path):
        df = collect_from_batch(pq_exists=False)
        print(f"\n[DEBUG] Data Collection completed: {len(df)} records fetched.")

        print(f"\n[DEBUG] Beginning attention-based extraction.")
        df = extract_attention_scores(df, top_n=5)
        print("\n[INFO] Attention-based keywords extracted.")

        save_partitioned_pq(df, keyword_path, rows_per_partition=5000)
        print(f"\n[DEBUG] Saved .parquet to {keyword_path}.")

    #if not os.listdir(keyword_path):
    #    raise FileNotFoundError(f"[ERROR] There really should be parquet files at this point.")

    all_files = [
        f for f in os.listdir(keyword_path) 
        if f.endswith('.parquet')
    ]
    if all_files:
        df = pd.read_parquet(os.path.join(tf_idf_path, all_files[0]))
        print(f"\n[INFO] Parquet data found. Loading from {PQ_PATH}")
        print(f"\n[INFO] these files should be preprocessed up to tf-idf vectorization.")
    else:
        print(f"\n[ERROR] No parquet files found in {PQ_PATH}.")
        stop_event.set()
        animation_thread.join()
        return

    stop_event.set()
    animation_thread.join()

    # Save processed data
    print(f"\n[INFO] Saving to {keyword_path}.")
    save_partitioned_pq(df, keyword_path, rows_per_partition=5000)
    print(f"\n[INFO] Complete.")

    # Print sample output
    print(df[['overview', 'keywords']].sample(5))

main()