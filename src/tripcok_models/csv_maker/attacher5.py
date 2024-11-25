# attacher5.py
import pandas as pd
import os
import time
import requests

CURRENT_SAVED = "saved_5.csv"
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 5     # secs

DETAIL_COMMON_URL = "http://apis.data.go.kr/B551011/KorService1/detailCommon1"
API_KEY = os.getenv("TOURISM_API")

def get_batch(df, n, batch_size=BATCH_SIZE):
    start_idx = n * batch_size
    end_idx = start_idx + batch_size
#    print(f"Batch {n}: {start_idx} to {end_idx}")  # Debug print to see the ranges

    if start_idx >= len(df):
        return pd.DataFrame()
    return df.iloc[start_idx:end_idx]


def load_columns(batch, current_batch):

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

                if not response.text.strip()
                    print(f"Empty response for contentid {contentid}")
                    break

                data = response.json()

                if 'response' in data and 'body' in data['response'] and 'items' in data['response']['body']:
                    items = data['response']['body']['items']['item']
                    if items:
                        item = items[0]
                        overview = item.get('overview', 'Unknown')
                        homepage = item.get('homepage', 'Unknown')
        
                        batch.at[index, 'overview'] = overview
                        batch.at[index, 'homepage'] = homepage
                        break
                    else:
                        print("No items found for contentid {contentid}")
                else:
                    print(f"Invalid response format for contentid {contentid}: {data}")
                    break
                    
            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"[ERROR] Failed to fetch details for contentid {contentid}: {e}")
                if retries < MAX_RETRIES:
                    print(f"Retrying... ({retries}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Max retries reached for contentid {contentid}. Skipping.")                    
                    batch.at[index, 'overview'] = overview
                    batch.at[index, 'homepage'] = homepage
                    break

    print(f"Completed: batch {current_batch} of size {BATCH_SIZE}")



def main():
    df = pd.read_csv(CURRENT_SAVED)
    df['overview'] = df['overview'].astype('object')
    df['homepage'] = df['homepage'].astype('object')

    current_batch = 0

    while True:
        batch = get_batch(df, current_batch)
        if batch.empty:
            break

        load_columns(batch, current_batch)
        batch.to_csv(f"batch/batch_{current_batch}.csv", index=False)        
        current_batch += 1
    
    df.to_csv("updated_save_5.csv", index=False)

main()