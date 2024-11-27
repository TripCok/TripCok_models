# t3.py

import requests
import csv
import os

def t3(page_no):
    LIST_URL = "http://apis.data.go.kr/B551011/KorService1/areaBasedList1"
    API_KEY = os.getenv("TOURISM_API")
    params = {
        "MobileOS": "ETC", "MobileApp": "tester",
        "_type": "json", "arrange": "A", 
        "numOfRows": 1000, "pageNo": page_no,
        "serviceKey": API_KEY
    }

# API에서 데이터 가져오기 try
    try:
        response = requests.get(LIST_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("[ERROR] When fetching page {page_no}: {e}")
        return [], -1

# 읽어온 데이터 parsing
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
        print(f"[ERROR] Skipping page...")
        return [], -1

    if curr_cnt < 1000 or page_no * 1000 >= total:
        return entries, -total
    return entries, total


# MAIN

# { 카테고리코드: 카테고리명 } dictionary 형성
translate_dict = {}
with open("translate.csv", "r") as t:
    reader = csv.reader(t)
    for row in reader:
        translate_dict[row[1]] = row[4]
        translate_dict[row[2]] = row[5]
        translate_dict[row[3]] = row[6]

print(translate_dict)

# data concat
full_file = []
page_no = 1
curr_cnt = 0

while True:
    entries, total = t3(page_no)

# 비정상종료
    if total == -1: break
    errors = 0

    if entries:
        for row in entries:
            try:        # 위의 translate idx 참조
                row[2] = translate_dict.get(row[2], 'Unknown')
                row[3] = translate_dict.get(row[3], 'Unknown')
                row[4] = translate_dict.get(row[4], 'Unknown')
            except KeyError as e:
                print(f"[ERROR] at {row}: {e}")
                errors += 1
                continue
            full_file.append(row)

        curr_cnt += (len(entries) - errors)
        print(f"Page {page_no} collected. {curr_cnt}/{total} items.")

# 정상종료
    if total < 0:
        break
    page_no += 1

# write to t3.csv

with open("t4.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(full_file)

print(f"Total_pages: {page_no}")

