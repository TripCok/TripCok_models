# kobret2.py

# 모델 트레이닝을 위한 라이브러리
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

# 로딩 애니메이션을 위한 라이브러리
import sys
import time
import threading

# batch 파일들의 저장 경로 - import 용
CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'
PQ_PATH = '/home/nishtala/TripCok/TripCok_models/src/tripcok_models/models/parquet/'

# stop_words: 수동으로 잡아낸 제거할 단어들
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

# tokenizer: koBERT 모델의 vocabulary 토크나이저
# 한국어 트레이닝이 되어 있으므로, 형태소-어미 분리 기능을 포함함
tokenizer = get_tokenizer()

# koBERT는 BERT 모델을 한국어 처리에 더 적합하게 변경한 라이브러리
# 즉, pre-trained
model = get_kobert_model()

# 모델을 학습이 아닌 평가(evaluation) 모드로 전환
# 즉, 학습한다면 사용하는 랜덤성 적용을 더이상 하지 않음
# 추후 kobert 자체에 추가적인 학습을 시키게 될 경우 변경될 수 있음
model.eval()


# 텍스트를 받으면 토큰(자연어처리의 기본유닛)으로 변환
def tokenize_korean(text: str):
    tokens = tokenizer.tokenize(text)
    return tokens


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


# koBERT의 tokenizer 에 내장된 한국어 형태소-어미 분리 기능을 통해
# overview 텍스트를 형태소 단위로 파편화 -> 추후 걸러낼 때 용이

def tokenize_overview(df: pd.DataFrame):
    # 문장부호 지우개
    translator = str.maketrans('', '', string.punctuation)

    # tokenizer 을 통해 형태소 단위로 분리된 텍스트 생성
    df['token_string'] = df['overview'].apply(
        lambda x: ' '.join(
            tokenizer.tokenize(
                re.sub(r'\d+', '', x.translate(translator))     # 문장부호 + 숫자 제거
            ) if isinstance(x, str) else []
        )
    )
    return df

# TF-IDF: Term Frequency - Inverse Document Frequency
# TF: 현 텍스트(current text - document)에서 자주 등장했을 때 점수가 높다 (고빈도)
# IDF: 전체 텍스트에서(full database - corpus) 자주 등장하지 않았을 때 점수가 높다 (유일성)
# 이 둘을 통해 유의미한 키워드를 결정짓는 방식

def apply_tf_idf(df: pd.DataFrame, top_n_words: int):
    tf_idf_vectorizer = TfidfVectorizer(max_features=top_n_words)
    tfidf_matrix = tf_idf_vectorizer.fit_transform(df['token_string'])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
         columns = tf_idf_vectorizer.get_feature_names_out()
    )

    df['tf_idf'] = tfidf_df.apply(
        lambda row: row.nlargest(top_n_words).index.tolist(), axis=1
    )

    return df.drop(columns = ['token_string'])


# 전처리 함수 두개 묶어서 메인에서 보기 편하도록
def preprocess_text(df: pd.DataFrame, top_n_words: int):
    df = tokenize_overview(df)
    df = apply_tf_idf(df, top_n_words)
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



def main():
    print("[DEBUG] Main function has started.\n")
    print("[DEBUG] Starting data collection.")

    tf_idf_path = PQ_PATH + "/tf-idf/"
    keyword_path = PQ_PATH + "/keywords/"

    # 경로 체크
    for path in [tf_idf_path, keyword_path]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"\n[INFO] path {path} created.")
        else:
            print(f"\n[INFO] path {path} found.")

    # 애니메이션 시작
    stop_event = threading.Event()
    animation_thread = threading.Thread(target=loading_animation, args=("Processing...", stop_event))
    animation_thread.start()

    # 전역변수: 우리 TF_IDF 처리해서 저장한 적 있나요?
    if not os.listdir(tf_idf_path):

        # 없으면 저 멀리 ../csv_maker/batches 에서 가져와
        df = collect_from_batch(pq_exists=False)
        print(f"\n[DEBUG] Data collection completed: {len(df)} records fetched.")
    
        # TF_IDF 전처리 후
        print("\n[DEBUG] Starting text preprocessing.")    
        df = preprocess_text(df, 100)
        print(f"\n[DEBUG] Text preprocessing completed. {len(df)} records processed.")

        # 저장해줌
        save_partitioned_pq(df, tf_idf_path, rows_per_partition=5000)
        print(f"\n[DEBUG] Saved TF-IDF preprocessed .parquet to {PQ_PATH}")

    if not os.listdir(tf_idf_path):
        raise FileNotFoundError(f"[ERROR] There really should be parquet files at this point.")

    all_files = [f for f in os.listdir(tf_idf_path) if f.endswith('.parquet')]
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

    print("\n[INFO] Printing preprocessed text list...")
    print(df[['title', 'tf_idf']].sample(7))

    return

main()