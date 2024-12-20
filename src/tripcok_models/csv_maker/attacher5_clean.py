import pandas as pd
import os
import time
import requests
import threading
import sys
from keylib import get_keys

# 전역 상수
CURRENT_SAVED = "saved_5.csv"
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 5
LAST_CONTENT_ID_FILE = "last_content_id.txt"  # 마지막 contentId를 저장할 파일
DETAIL_COMMON_URL = "http://apis.data.go.kr/B551011/KorService1/detailCommon1"

stop_loading = False

# 로딩 애니메이션
def loading_animation(message):
    symbols = ["\\", "|", "/", "-"]
    idx = 0
    while not stop_loading:
        sys.stdout.write(f"\r{message} {symbols[idx]}")
        sys.stdout.flush()
        idx = (idx + 1) % len(symbols)
        time.sleep(0.2)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")

# 마지막 contentId 읽기
def get_last_content_id():
    if os.path.exists(LAST_CONTENT_ID_FILE):
        with open(LAST_CONTENT_ID_FILE, 'r') as f:
            return f.read().strip()
    return None

# 마지막 contentId 저장
def save_last_content_id(contentid):
    with open(LAST_CONTENT_ID_FILE, 'w') as f:
        f.write(str(contentid))

# 배치 가져오기
def get_batch(df, start_content_id=None, batch_size=BATCH_SIZE):
    if start_content_id:
        # start_content_id가 데이터프레임에 존재하는지 확인
        matching_rows = df[df['contentid'] == start_content_id]
        if matching_rows.empty:
            print(f"[WARNING] start_content_id '{start_content_id}' not found in the dataset. Starting from the beginning.")
            start_idx = 0
        else:
            start_idx = matching_rows.index[0]
    else:
        start_idx = 0

    end_idx = start_idx + batch_size
    if start_idx >= len(df):
        return pd.DataFrame()
    return df.iloc[start_idx:end_idx]

# 데이터 요청
def fetch_data(contentid, key):
    params = {
        "MobileOS": "ETC",
        "MobileApp": "tester",
        "_type": "json",
        "contentId": contentid,
        "defaultYN": "Y",
        "overviewYN": "Y",
        "serviceKey": key
    }
    for attempt in range(MAX_RETRIES):
        time.sleep(1)
        response = requests.get(DETAIL_COMMON_URL, params=params)
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                print(f"[ERROR] Non-JSON response for contentid {contentid}: {response.text.strip()[:100]}")
                break
        else:
            print(f"[ERROR] HTTP {response.status_code} for contentid {contentid}. Retrying... ({attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
    return None

# API 응답 데이터 처리
def process_data(data, contentid):
    if not isinstance(data, dict):
        print(f"[ERROR] Unexpected data format for contentid {contentid}")
        return 'Unknown', 'Unknown'
    
    response_data = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if response_data:
        item = response_data[0]
        overview = item.get('overview', 'Unknown')
        homepage = item.get('homepage', 'Unknown')
        return overview, homepage
    else:
        print(f"No items found for contentid {contentid}")
        return 'Unknown', 'Unknown'

# 배치 처리
def process_batch(batch):
    global stop_loading
    updates = []

    for index, row in batch.iterrows():
        contentid = row['contentid']
        key = get_keys()

        print(f"Using API key: {key}")
        data = fetch_data(contentid, key)
        print(data)
        if data:
            overview, homepage = process_data(data, contentid)
        else:
            overview, homepage = 'Unknown', 'Unknown'
        
        updates.append((index, overview, homepage))
        save_last_content_id(contentid)  # 처리한 contentId 저장

    stop_loading = True

    # 업데이트된 데이터 저장
    for index, overview, homepage in updates:
        batch.at[index, 'overview'] = overview
        batch.at[index, 'homepage'] = homepage

    return batch

# 전체 데이터 처리
def process_all_batches(df, start_content_id=None):
    global stop_loading
    batch = get_batch(df, start_content_id)

    while not batch.empty:
        stop_loading = False
        animation_thread = threading.Thread(
            target=loading_animation,
            args=(f"Processing batch starting from contentId {start_content_id or 'first'}...",)
        )
        animation_thread.start()

        batch = process_batch(batch)
        animation_thread.join()

        start_content_id = batch.iloc[-1]['contentid']  # 마지막 contentId로 갱신
        batch = get_batch(df, start_content_id)

# 메인 함수
def main():
    try:
        df = pd.read_csv(CURRENT_SAVED)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {CURRENT_SAVED}")
        return

    df['overview'] = df['overview'].astype('object')
    df['homepage'] = df['homepage'].astype('object')

    start_content_id = get_last_content_id()  # 마지막 contentId 가져오기
    process_all_batches(df, start_content_id)

    df.to_csv("updated_save_5.csv", index=False)

# 실행
if __name__ == "__main__":
    main()
