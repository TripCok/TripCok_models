# attacher5.py
import pandas as pd
import os
import time
import requests
import threading
import sys

CURRENT_SAVED = "saved_5.csv"
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 5     # secs
START_BATCH = 9     # on fail, retry from last checkpoint by changing this

DETAIL_COMMON_URL = "http://apis.data.go.kr/B551011/KorService1/detailCommon1"
API_KEY = os.getenv("TOURISM_API")


if not API_KEY:
    raise ValueError("\n[ERROR] API KEY missing.")

stop_event = threading.Event()

# animations for load
def loading_animation(message, stop_event):
    symbols = ["\\", "|", "/", "-"]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {symbols[idx]}")     # Overwrite the line
        sys.stdout.flush()
        idx = (idx + 1) % len(symbols)                      # Cycle through symbols
        time.sleep(0.2)                                     # animation speed
    sys.stdout.write("\r" + " "*(len(message) + 2) + "\r")  # Clear the line


# returns slice: change in slice affects original list
def get_batch(df, n, batch_size=BATCH_SIZE):
    start_idx = n * batch_size
    end_idx = start_idx + batch_size
    if start_idx >= len(df):
        return pd.DataFrame()
    return df.iloc[start_idx:end_idx]


def load_columns(batch, current_batch):
    updates = []
    for index, row in batch.iterrows():
        contentid = row['contentid']
        params = {
            "MobileOS": "ETC",
            "MobileApp": "tester",
            "_type": "json",
            "contentId": contentid,
            "defaultYN": "Y",
            "overviewYN": "Y",
            "serviceKey": API_KEY,
        }

        overview, homepage = 'Unknown', 'Unknown'
        retries = 0

        while retries < MAX_RETRIES:
            try:   
                response = requests.get(DETAIL_COMMON_URL, params=params)
                response.raise_for_status()

                try:
                    data = response.json()
                    if not isinstance(data, dict):
                        raise ValueError(f"Unexpected response format for data: {data}")
                except ValueError:
                    print(f"\n[ERROR] Non-JSON response for contentid {contentid}: {response.text.strip()[:100]}")
                    break                

                # Extract response data
                if isinstance(data, dict):
                    response_data = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                    if response_data:
                        item = response_data[0]
                        if not isinstance(item, dict):
                            print(f"[ERROR] Unexpected item format: {item}")
                            continue
                        # update values
                        overview = item.get('overview', 'Unknown')
                        homepage = item.get('homepage', 'Unknown')
                        break
                    else:
                        print(f"[INFO] No data found for contentid {contentid}. Skipping.")
                        break
                else:
                    print(f"[ERROR] Unexpected data format for contentid {contentid}: {data}")
                    break    
            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"\n[ERROR] Failed to fetch details for contentid {contentid}: {e}")
                if retries < MAX_RETRIES:
                    print(f"Retrying... ({retries}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"\nMax retries reached for contentid {contentid}. Skipping.")                    
                    break

        updates.append((index, overview, homepage))
    
    print()
    print(f"Completed: batch {current_batch} of size {BATCH_SIZE}")
    
    for index, overview, homepage in updates:
        batch.loc[index, ['overview', 'homepage']] = [overview, homepage]
    
    if len(updates) == len(batch):
        batch.to_csv(f"batch/batch_{current_batch}.csv", index=False)        
        print(f"Saved: batch_{current_batch}.csv\n")
    else:
        print(f"[WARNING] Partial data for batch {current_batch} was not saved.")


# batch 1개만 테스트 해볼때: START_BATCH 만 시도
def single_try():
    try:
        df = pd.read_csv(CURRENT_SAVED)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {CURRENT_SAVED}")
        return

    df['overview'] = df['overview'].astype('object')
    df['homepage'] = df['homepage'].astype('object')

    os.makedirs("batch", exist_ok=True)
    current_batch = START_BATCH

    batch = get_batch(df, current_batch)
    if batch.empty:
        return

    animation_thread = threading.Thread(
        target=loading_animation, 
        args=(f"Processing batch {current_batch} of size {BATCH_SIZE}...", stop_event)
    )
    animation_thread.start()

    try:
        load_columns(batch, current_batch)
    except Exception as e:
        print(f"[ERROR] An unexpected error occured in main: {e}")
    finally:
        stop_event.set()
        animation_thread.join()


# START_BATCH 부터 쭉 시도
def main():
    try:
        df = pd.read_csv(CURRENT_SAVED)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {CURRENT_SAVED}")
        return

    df['overview'] = df['overview'].astype('object')
    df['homepage'] = df['homepage'].astype('object')

    os.makedirs("batch", exist_ok=True)
    current_batch = START_BATCH

    while True:
        batch = get_batch(df, current_batch)
        if batch.empty:
            break

        stop_loading = False
        animation_thread = threading.Thread(
            target=loading_animation, 
            args=(f"Processing batch {current_batch} of size {BATCH_SIZE}...", stop_event)
        )
        animation_thread.start()

        try:
            load_columns(batch, current_batch)
        except Exception as e:
            print(f"[ERROR] An unexpected error occured in main: {e}")
        finally:
            stop_event.set()
            animation_thread.join()

        current_batch += 1
    


main()
#single_try()