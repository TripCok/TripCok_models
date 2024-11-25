import requests
import csv

def t1(page_no):
    LIST_URL = "http://apis.data.go.kr/B551011/KorService1/areaBasedList1"
    API_KEY = "CbGTCUMxSxMeDRbn8OvoS2LOss5VQnLpGlSdTAcErvIP2ly2oBtr5EyTPNgp3Di7HSdJKqAZYLwwdo5qSBkvxg=="

    params = {
        "MobileOS": "WIN",
        "MobileApp": "tester",
        "_type": "json",
        "arrange": "A",
        "numOfRows": 1000,
        "pageNo": page_no,
        #"contentTypeId": 12, 
        "serviceKey": API_KEY
# 12:관광지, 14:문화시설, 15:축제공연행사, 25:여행스포츠, 28:레포츠, 32:숙박, 38:쇼핑, 39:음식점
}

    try:
        response = requests.get(LIST_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[ERROR] Page {page_no}: {e}")
        return [], False

    try:
        items = data['response']['body']['items']['item']
        listed_categories = [[
            item.get('title', 'Unknown'), 
            item.get('cat1', 'Unknown'), 
            item.get('cat2', 'Unknown'), 
            item.get('cat3', 'Unknown'),
            item.get('addr1', '').split()[0] if item.get('addr1') else 'Unknown'
        ] for item in items ]

        current_count = len(items)
        total_count = data['response']['body']['totalCount']
    except KeyError:
        print(f"[ERROR] Unexpected response structure on page {page_no}.")
        return [], False

    print(f"Page {page_no} collected. {page_no * len(listed_categories)}/{total_count} items.")

    if current_count < 1000 or page_no * 1000 >= total_count:
        return listed_categories, False
    return listed_categories, True

# MAIN
page_no = 1
with open("t1_result.csv", "w") as f:
    writer = csv.writer(f)
    while True:
        listed_categories, cont = t1(page_no)
        
        # debug line: check format
        if page_no == 1: 
            print(listed_categories[0])
    
        if not cont: break
        page_no += 1

        if listed_categories:
            writer.writerow(listed_categories)

print(f"Total pages: {page_no}")
