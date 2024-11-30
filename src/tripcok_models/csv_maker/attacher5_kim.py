import pandas as pd
import os
import time
import requests
import threading
import sys
from keylib import get_keys

# 처리할 데이터 파일 이름
CURRENT_SAVED = "saved_5.csv"

# 배치 크기 및 재시도 설정
BATCH_SIZE = 100            # 한 번에 처리할 데이터의 크기
MAX_RETRIES = 3             # 요청 실패 시 최대 재시도 횟수
RETRY_DELAY = 5             # 재시도 간 대기 시간 (초 단위)
START_BATCH = 185             # 처리 시작할 배치 번호 (재시작 시 유용)

# API 호출 URL
DETAIL_COMMON_URL = "http://apis.data.go.kr/B551011/KorService1/detailCommon1"

# 애니메이션 중단 플래그
stop_loading = False

# 로딩 애니메이션 함수 (콘솔에 로딩 표시)
def loading_animation(message):
    symbols = ["\\", "|", "/", "-"]  # 순환할 로딩 기호
    idx = 0
    while not stop_loading:  # stop_loading 플래그가 False일 동안 애니메이션 실행
        sys.stdout.write(f"\r{message} {symbols[idx]}")  # 같은 줄에 덮어씌우기
        sys.stdout.flush()
        idx = (idx + 1) % len(symbols)  # 기호 순환
        time.sleep(0.2)  # 애니메이션 속도 제어
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")  # 애니메이션 종료 시 콘솔 정리


def get_batch(df, n, batch_size=BATCH_SIZE):
    start_idx = n * batch_size  # 시작 인덱스 계산
    end_idx = start_idx + batch_size  # 종료 인덱스 계산
    if start_idx >= len(df):  # 시작 인덱스가 데이터 크기를 넘어가면 빈 데이터 반환
        return pd.DataFrame()
    return df.iloc[start_idx:end_idx]  # 지정된 배치 반환


# 배치 데이터를 처리하고 결과를 저장하는 함수
def load_columns(batch, current_batch):
    global stop_loading

    updates = []  # 처리 결과를 저장할 리스트
    non_updates = []
    # 배치 내의 각 행을 처리
    for index, row in batch.iterrows():
        contentid = row['contentid']  # 데이터의 contentid 값 가져오기
        key = get_keys()
        
        params = {
            "MobileOS": "ETC",
            "MobileApp": "tester",
            "_type": "json",
            "contentId": contentid,
            "defaultYN": "Y",
            "overviewYN": "Y",
            "serviceKey": key  # API 키를 직접 삽입
        }
        
        print(key)

        overview, homepage = 'Unknown', 'Unknown'  # 기본 값 설정
        retries = 0  # 재시도 횟수 초기화

        # API 호출 및 재시도 처리
        while retries < MAX_RETRIES:
            time.sleep(1)
            try:
                response = requests.get(DETAIL_COMMON_URL, params=params)  # API 호출
                response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

                try:
                    data = response.json()  # JSON 응답 파싱
                except ValueError:  # JSON 파싱 실패
                    print(f"\n[ERROR] Non-JSON response for contentid {contentid}: {response.text.strip()[:100]}")
                    break

                if isinstance(data, dict):
                    response_data = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                    print(response_data)
                    if response_data:
                        item = response_data[0]
                        overview = item.get('overview', 'Unknown')
                        homepage = item.get('homepage', 'Unknown')
                    break  # 성공적으로 데이터를 처리한 경우 반복 종료
                else:
                    print(f"[ERROR] Unexpected data format for contentid {contentid}: {type(data)}")
                    break

#                response_data = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
#                print(response_data)
#                if response_data:
#                    item = response_data[0]
#                    # 추출한 데이터로 업데이트
#                    overview = item.get('overview', 'Unknown')
#                    homepage = item.get('homepage', 'Unknown')
#                    break
#                else:  # 응답 데이터가 없는 경우
#                    print(f"No items found for contentid {contentid}")
#                    break

            except requests.exceptions.RequestException as e:  # 요청 실패
                retries += 1
                print(f"\n[ERROR] Failed to fetch details for contentid {contentid}: {e}")
                if retries < MAX_RETRIES:  # 최대 재시도 횟수 이하일 경우 대기 후 재시도
                    print(f"Retrying... ({retries}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
                else:  # 재시도 횟수 초과 시 중단
                    print(f"\nMax retries reached for contentid {contentid}. Skipping.")
                    break

        updates.append((index, overview, homepage))  # 처리된 데이터 저장

    stop_loading = True  # 로딩 애니메이션 중단

    print(f"Completed: batch {current_batch} of size {BATCH_SIZE}")  # 배치 처리 완료 메시지 출력

    # 배치 내 각 행을 업데이트
    for index, overview, homepage in updates:
        batch.at[index, 'overview'] = overview
        batch.at[index, 'homepage'] = homepage

    # 처리된 배치를 파일로 저장
    batch.to_csv(f"batch/batch_{current_batch}.csv", index=False)
    print(f"Saved: batch_{current_batch}.csv\n")


# 메인 실행 함수
def main():
    global stop_loading

    # 입력 데이터 파일 로드
    try:
        df = pd.read_csv(CURRENT_SAVED)
    except FileNotFoundError:  # 파일이 없으면 에러 메시지 출력
        print(f"[ERROR] File not found: {CURRENT_SAVED}")
        return

    # 데이터프레임에 새 컬럼 추가 및 타입 설정
    df['overview'] = df['overview'].astype('object')
    df['homepage'] = df['homepage'].astype('object')

    # 배치 저장 폴더 생성
    os.makedirs("batch", exist_ok=True)
    current_batch = START_BATCH  # 처리 시작할 배치 설정

    # 배치 반복 처리
    while True:
        batch = get_batch(df, current_batch)  # 현재 배치 데이터 가져오기
        if batch.empty:  # 배치가 비어 있으면 종료
            break

        # 로딩 애니메이션 시작
        stop_loading = False
        animation_thread = threading.Thread(target=loading_animation, args=(f"Processing batch {current_batch} of size {BATCH_SIZE}...",))
        animation_thread.start()

        # 배치 데이터 처리
        load_columns(batch, current_batch)
        # 로딩 애니메이션 중단 및 정리
        animation_thread.join()

        current_batch += 1  # 다음 배치로 이동

    # 전체 데이터 저장
    df.to_csv("updated_save_5.csv", index=False)

# 메인 함수 실행
main()
