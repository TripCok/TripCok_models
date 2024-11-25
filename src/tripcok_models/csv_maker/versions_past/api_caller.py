import requests

def call_names(page_no):
    API_KEY = "CbGTCUMxSxMeDRbn8OvoS2LOss5VQnLpGlSdTAcErvIP2ly2oBtr5EyTPNgp3Di7HSdJKqAZYLwwdo5qSBkvxg=="
    BASE_URL = "http://apis.data.go.kr/B551011/KorService1/areaBasedList1"

    params = {
        "serviceKey": API_KEY,
        "numOfRows": 1000,  # 최대 1000개의 데이터를 가져옴
        "pageNo": page_no,
        "MobileOS": "ETC",
        "MobileApp": "AppTest",
        "_type": "json"
    }
        
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[ERROR] Page {page_no}: {e}")
        return [], False

    try:
        items = data['response']['body']['items']['item']
        travel_names = [item['title'] for item in data['response']['body']['items']['item']]
        current_count = len(items)
        total_count = data['response']['body']['totalCount']
    except KeyError:
        print(f"[ERROR] Unexpected response structure on page {page_no}.")
        return [], False

    print(f"Page {page_no} collected. {page_no * len(travel_names)}/{total_count} items.")

    if current_count < 1000 or page_no * 1000 >= total_count:
        return travel_names, False
    return travel_names, True

with open("lists.csv", "w") as f:
    page_no = 1
    unique_travel_names = set()

    while True:
        travel_names, cont = call_names(page_no)    
        unique_travel_names.update(travel_names)

        if not cont: 
            break
        page_no += 1
    f.write(",".join(unique_travel_names))

    print(f"Total pages: {page_no}")
    print(f"Total unique travel destinations collected: {len(unique_travel_names)}")


