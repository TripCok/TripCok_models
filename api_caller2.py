import requests
import json

# API 키와 기본 URL
API_KEY = 'CbGTCUMxSxMeDRbn8OvoS2LOss5VQnLpGlSdTAcErvIP2ly2oBtr5EyTPNgp3Di7HSdJKqAZYLwwdo5qSBkvxg=='  # 여기에 발급받은 API 키 입력
BASE_URL = 'http://api.visitkorea.or.kr/openapi/service/rest/KorService/'

# 카테고리 정보와 관광지 목록을 가져오는 함수
def get_tourist_spots(category_code, num_of_rows=10, total_pages=5):
    all_spots = []  # 모든 관광지 정보를 담을 리스트
    
    for page_no in range(1, total_pages + 1):  # 여러 페이지를 가져오기 위해 반복
        # API 호출 URL
        url = f"{BASE_URL}areaBasedList"
        
        # 요청 파라미터
        params = {
            'serviceKey': API_KEY,  # API 키
            'numOfRows': num_of_rows,  # 한 번에 가져올 데이터 개수
            'pageNo': page_no,  # 페이지 번호
            'cat1': category_code,  # 카테고리 코드
            '_type': 'json'  # 응답 형식 (JSON)
        }
        
        # API 요청
        response = requests.get(url, params=params)
        
        # 응답 처리
        if response.status_code == 200:
            data = response.json()  # JSON 응답을 파싱
            if 'response' in data and 'body' in data['response']:
                spots = data['response']['body']['items']['item']
                all_spots.extend(spots)  # 각 페이지에서 받은 관광지 목록을 추가
                print(f"Page {page_no} 데이터 추가됨.")
            else:
                print("응답 데이터에 오류가 있습니다.")
        else:
            print(f"API 요청 실패: {response.status_code}")
    
    return all_spots  # 전체 관광지 목록 반환

# 관광지 정보를 JSON 파일로 저장하는 함수
def save_to_json(data, filename='tourist_spots.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"데이터가 {filename} 파일에 저장되었습니다.")

# 예시: 'C01' (전체) 카테고리로 관광지 가져오기
category_code = 'C01'  # 카테고리 코드 예시
total_pages = 5  # 총 5페이지의 데이터를 가져오겠다고 설정 (필요에 따라 변경)

tourist_spots = get_tourist_spots(category_code, total_pages=total_pages)

if tourist_spots:
    save_to_json(tourist_spots)  # 관광지 데이터를 JSON 파일로 저장

