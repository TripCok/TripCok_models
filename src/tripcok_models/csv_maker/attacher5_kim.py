# attacher5.py
import pandas as pd
import os
import time
import requests
import threading
import sys
from keylibs import RoundRobin, get_keys

CURRENT_SAVED = "saved_5.csv"
BATCH_SIZE = 500
MAX_RETRIES = 3
RETRY_DELAY = 5     # secs
START_BATCH = 0     # on fail, retry from last checkpoint by changing this

DETAIL_COMMON_URL = "http://apis.data.go.kr/B551011/KorService1/detailCommon1"

keyManger = RoundRobin()

for 

if not keyManager.get_keys()
    raise ValueError("\n[ERROR] API KEY missing.")

stop_loading = False

# animations for load
def loading_animation(message):
    symbols = ["\\", "|", "/", "-"]
    idx = 0
    while not stop_loading:
        sys.stdout.write(f"\r{message} {symbols[idx]}")  # Overwrite the line
        sys.stdout.flush()
        idx = (idx + 1) % len(symbols)  # Cycle through symbols
        time.sleep(0.2)  # Control speed of the animation
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")  # Clear the line


# returns slice: change in slice affects original list
def get_batch(df, n, batch_size=BATCH_SIZE):
    start_idx = n * batch_size
    end_idx = start_idx + batch_size
    if start_idx >= len(df):
        return pd.DataFrame()
    return df.iloc[start_idx:end_idx]


def load_columns(batch, current_batch):
    global stop_loading

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
                except ValueError:
                    print(f"\n[ERROR] Non-JSON response for contentid {contentid}: {response.text.strip()[:100]}")
                    break                

                # Extract response data
                response_data = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                if response_data:
                    item = response_data[0]
                    # update values
                    overview = item.get('overview', 'Unknown')
                    homepage = item.get('homepage', 'Unknown')
                    break
                else:
                    print("No items found for contentid {contentid}")
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

    stop_loading = True
    print()

    print(f"Completed: batch {current_batch} of size {BATCH_SIZE}")
    
    for index, overview, homepage in updates:
        batch.at[index, 'overview'] = overview
        batch.at[index, 'homepage'] = homepage
    
    batch.to_csv(f"batch/batch_{current_batch}.csv", index=False)        
    print(f"Saved: batch_{current_batch}.csv\n")



def main():
    global stop_loading

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
        animation_thread = threading.Thread(target=loading_animation, args=(f"Processing batch {current_batch} of size {BATCH_SIZE}...",))
        animation_thread.start()

        load_columns(batch, current_batch)
        animation_thread.join()

        current_batch += 1
    
    df.to_csv("updated_save_5.csv", index=False)

main()
