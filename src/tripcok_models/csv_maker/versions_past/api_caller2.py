import requests

# Base URLs
LIST_URL = "http://apis.data.go.kr/B551011/LocgoHubTarService/areaBasedList"
DETAIL_URL = "http://apis.data.go.kr/B551011/LocgoHubTarService/detailCommon"

# API Key
API_KEY = "CbGTCUMxSxMeDRbn8OvoS2LOss5VQnLpGlSdTAcErvIP2ly2oBtr5EyTPNgp3Di7HSdJKqAZYLwwdo5qSBkvxg=="

# Step 1: Fetch a list of tourist attractions
params_list = {
    "serviceKey": API_KEY,
    "MobileOS": "ETC",
    "MobileApp": "TourApp",
    "areaCode": "1",  # Example area code
    "numOfRows": 10,
    "pageNo": 1,
    "_type": "json"
}
response = requests.get(LIST_URL, params=params_list)
data = response.json()

# Extract contentId
for item in data["response"]["body"]["items"]["item"]:
    content_id = item["contentid"]
    print(f"Fetching details for contentId: {content_id}")
    
    # Step 2: Fetch detailed information
    params_detail = {
        "serviceKey": API_KEY,
        "MobileOS": "ETC",
        "MobileApp": "TourApp",
        "contentId": content_id,
        "_type": "json"
    }
    detail_response = requests.get(DETAIL_URL, params=params_detail)
    detail_data = detail_response.json()
    
    # Parse and display the detailed data
    print(detail_data)

