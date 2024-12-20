import requests
import csv
import os

def t2(page_no):
    LIST_URL = "http://apis.data.go.kr/B551011/KorService1/areaBasedList1"
#    API_KEY = "CbGTCUMxSxMeDRbn8OvoS2LOss5VQnLpGlSdTAcErvIP2ly2oBtr5EyTPNgp3Di7HSdJKqAZYLwwdo5qSBkvxg=="
    API_KEY = os.getenv("TOURISM_API")
    params = {
        "MobileOS": "WIN",
        "MobileApp": "tester",
        "_type": "json",
        "arrange": "A",
        "numOfRows": 1000,
        "pageNo": page_no,
        "serviceKey": API_KEY
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
            item.get('addr1', '').split()[0] if item.get('addr1') else 'NULL'
        ] for item in items ]

        current_count = len(items)
        total_count = data['response']['body']['totalCount']
    except KeyError:
        print(f"[ERROR] Unexpected response structure on page {page_no}.")
        return [], False

    if len(listed_categories) == 1000:
        acc_sum = page_no * 1000
    else:
        acc_sum = (page_no -1) * 1000 + len(listed_categories)

    print(f"Page {page_no} collected. {acc_sum}/{total_count} items.")

    if current_count < 1000 or page_no * 1000 >= total_count:
        return listed_categories, False
    return listed_categories, True

# MAIN
page_no = 1
translate_dict = {}
full_file = []

with open("translate.csv", "r") as t:
    reader = csv.reader(t)
    for row in reader:
        translate_dict[row[1]] = row[4]
        translate_dict[row[2]] = row[5]
        translate_dict[row[3]] = row[6]

while True:
    listed_categories, cont = t2(page_no)
    if not cont: break
    page_no += 1

    if listed_categories:
        for row in listed_categories:
            try:
                row[1] = translate_dict[row[1]]
                row[2] = translate_dict[row[2]]
                row[3] = translate_dict[row[3]]
            except KeyError as e:
                print(f"[ERROR] at {row}: e")
                continue

            full_file.append(row)

with open("t2_result.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(full_file)

print(f"Total pages: {page_no}")



