# paginator_5.py

import requests
import csv
import os
import pandas as pd

TRANSLATE_FILE = "translate.csv"
SAVE_NAME = "saved_5"
SAVE_FORMAT = ".csv"

BATCH_SIZE = 10000       # numofrows, size of entries

AREA_BASED_URL = "http://apis.data.go.kr/B551011/KorService1/areaBasedList1"
#DETAIL_COMMON_URL = "http://apis.data.go.kr/B551011/KorService1/detailCommon1"
API_KEY = os.getenv("TOURISM_API")


TRANSLATE_DICT = {}

def load_translator():
    global TRANSLATE_DICT
    with open("translate.csv", "r") as t:
        reader = csv.reader(t)
        for row in reader:
            TRANSLATE_DICT[row[1]] = row[4]
            TRANSLATE_DICT[row[2]] = row[5]
            TRANSLATE_DICT[row[3]] = row[6]

# areaBasedList1 API 불러오는 함수
def area_based_API(page_no):
    params = {
        "MobileOS": "ETC", 
        "MobileApp": "tester",
        "_type": "json", 
        "arrange": "A", 
        "numOfRows": BATCH_SIZE, 
        "pageNo": page_no,
        "serviceKey": API_KEY
    }
    try:
        response = requests.get(AREA_BASED_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[ERROR] When fetching page {page_no}: {e}")
        return None, -1

    try:
        items = data['response']['body']['items']['item']
        entries = [[
            item.get('title', 'Unknown'),           # 여행지 지역명
            item.get('contentid', 'Unknown'),       # 여행지 contentid
            item.get('cat1', 'Unknown'),            # 대분류 코드, idx 2    > translate 할 때
            item.get('cat2', 'Unknown'),            # 중분류 코드, idx 3
            item.get('cat3', 'Unknown'),            # 소분류 코드, idx 4
            item.get('firstimage', 'Unknown'),       # 대표이미지
            item.get('addr1', '').split()[0] if item.get('addr1') else 'Unknown',   # 지역명 ex. 충청남도
            item.get('mapx', 'Unknown'),            # 좌표x
            item.get('mapy', 'Unknown')             # 좌표y
        ] for item in items ]

        curr_cnt = len(items)       # 읽어들인 총 엔트리(아이템) 수
        total = data['response']['body']['totalCount']

    except KeyError:
        print(f"[ERROR] KeyError on page {page_no}")
        return None, -1

    return entries, total

#def detail_common_API(contentid):
#    params = {
#        "MobileOS": "ETC",
#        "MobileApp": "tester",
#        "_type": "json",
#        "contentId": contentid,
#        "defaultYN": "Y",
#        "overviewYN": "Y",
#        "serviceKey": API_KEY,
#    }
#    try:
#        response = requests.get(DETAIL_COMMON_URL, params=params)
#        response.raise_for_status()
#        data = response.json()
#        item = data['response']['body']['items']['item'][0]
#
#        overview = item.get('overview', 'Unknown')
#        homepage = item.get('homepage', 'Unknown')
#        return overview, homepage
#    except Exception as e:
#        print(f"[ERROR] Failed to fetch details for contentid {contentid}: {e}") 
#        return None, None   

# cat1, cat2, cat3 코드를 카테고리명으로 번역하는 함수
# 리턴값은 에러가 생긴 횟수 (있다면)
def translate(entries):
    global TRANSLATE_DICT
    errors = 0
    for i in range(len(entries)):
        try:
            entries[i][2] = TRANSLATE_DICT.get(entries[i][2], 'Unknown')
            entries[i][3] = TRANSLATE_DICT.get(entries[i][3], 'Unknown')
            entries[i][4] = TRANSLATE_DICT.get(entries[i][4], 'Unknown')
        except KeyError as e:
            print(f"[ERROR] at {row}: {e}")
            errors += 1
    return errors

# WRITE 함수
def export(dataframe, name=SAVE_NAME, format=SAVE_FORMAT):
    output_file = name + format
    if format == ".csv":
        dataframe.to_csv(output_file, index=False, encoding="utf-8")
    elif format == ".json":
        dataframe.to_json(output_file, orient="records", lines=True, encoding="utf-8")
    elif format == ".parquet":
        dataframe.to_parquet(output_file, index=False)
    else:
        print(f"[ERROR] unrecognized format: {output_file}. File is not saved.")
        return
    print(f"Data exported to {output_file}")

# main
def main():
    
    load_translator()
    columns = [
        # area_based_API 
        "title", "contentid", "cat1", "cat2", "cat3", 
        "firstimage", "region", "mapx", "mapy",
        # detail_common_API
        "overview", "homepage", 
    ]
    full_data = pd.DataFrame(columns=columns)

    # pagination
    page_no, curr_cnt = 1, 0
    full_data_list = []

    while True:
        entries, total = area_based_API(page_no)
        if total == -1: break

        if entries:
            errors = translate(entries)     # translates & returns errornum
            
            for entry in entries:           # extends for overview and homepage
#                contentid = entry[1]
#                overview, homepage = detail_common_API(contentid)
#                entry.extend([overview, homepage])
                entry.extend([pd.NA, pd.NA])

            batch_df = pd.DataFrame(entries, columns=columns)
            full_data_list.append(batch_df)
    #        full_data = pd.concat([full_data, batch_df], ignore_index=True)
    
            curr_cnt += len(entries) - errors
            print(f"Page {page_no} collected. {curr_cnt}/{total} items.")

        if total < 0 or curr_cnt >= total:
            break
        page_no += 1

    full_data = pd.concat(full_data_list, ignore_index=True)

    print(f"Total collected entries: {len(full_data)}")
    export(full_data)


main()

