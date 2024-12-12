import pandas as pd
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
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

tokenizer = Okt()   # 토크나이저


# 텍스트를 받으면 토큰(자연어처리의 기본유닛)으로 변환
def tokenize_korean(text: str):
    tokens = tokenizer.morphs(text) if isinstance(text, str) else []
    return [
        token for token in tokens
        if token not in stop_words_kr
    ]


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


def tokenize_overview(df: pd.DataFrame):
    translator = str.maketrans('', '', string.punctuation)
    df['token_string'] = df['overview'].apply(
        lambda x: ' '.join(
            tokenize_korean(
                re.sub(r'\d+', '', str(x).translate(translator))
            )
        ) if isinstance(x, str) else ''
    )
    return df


def apply_tf_idf(df: pd.DataFrame, top_n_words: int):
    tf_idf_vectorizer = TfidfVectorizer(max_features=top_n_words)
    tfidf_matrix = tf_idf_vectorizer.fit_transform(df['token_string'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())

    df['tf_idf'] = tfidf_df.apply(
        lambda row: row.nlargest(top_n_words).index.tolist(), axis=1
    )
    return df.drop(columns=['token_string'])


def preprocess_text(df: pd.DataFrame, top_n_words: int):
    df = tokenize_overview(df)
    df = apply_tf_idf(df, top_n_words)
    return df


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


def extract_keywords_from_df(df: pd.DataFrame, top_n: int):
    total_records = len(df)

    def extraction(text: str, idx: int):
        if idx % 500 == 0 and idx > 0:
            print(f"\n[DEBUG] Processed {idx} records out of {total_records}")
        return extract_keywords(' '.join(text), top_n=top_n)

    df['keywords'] = [
        extraction(text, idx)
        for idx, text in enumerate(df['overview'])
    ]
    df['keywords'] = df.apply(
        lambda row: row['keywords'] + [row['cat3'], row['region']], axis=1
    )
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
    for i in range(0, len(df), rows_per_partition):
        partition_df = df.iloc[i:i+rows_per_partition]
        partition_df.to_parquet(os.path.join(output_path, f"partition_{i//rows_per_partition +1}.parquet"))
    

def main():
    print("[INFO] Starting preprocessing with KoNLPy.\n")
    
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

    if not os.listdir(tf_idf_path):        
        df = collect_from_batch(pq_exists=False)
        print(f"[INFO] Data loaded: {len(df)} records.")
        df = preprocess_text(df, top_n_words=100)
        save_partitioned_pq(df, tf_idf_path, rows_per_partition=5000)
        print("[INFO] Preprocessing completed and saved.\n")
    
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

    print()
    print("\n[INFO] Preprocessed data loaded! extracting keywords now...")
    print()

    df = extract_keywords_from_df(df, top_n=50)
    print(f"\n[DEBUG] Keyword extraction completed. {len(df)} keywords extracted.")

    save_partitioned_pq(df, keyword_path, rows_per_partition=5000)


    stop_event.set()
    animation_thread.join()

#    print("\n[INFO] Printing preprocessed text list...")
#    print(df[['title', 'tf_idf']].sample(7))

    return

main()