import requests


def standardize_address(data):
    # 주요 필드 정의
    province = data.get('province', '')
    city = data.get('city', '')
    borough = data.get('borough', '')
    road = data.get('road', '')
    house_number = data.get('house_number', '')
    village = data.get('village', '')
    town = data.get('town', '')
    county = data.get('county', '')

    # 표준화된 주소 조합
    standardized = f"{province} {city} {borough} {county} {town} {village} {road} {house_number}".strip()

    # 중복된 공백 제거
    standardized = ' '.join(standardized.split())
    return standardized


def reverse_geocode(lat, lng):
    if not lat or not lng:  # 빈 값 확인
        return "좌표 없음"

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": float(lat),  # 위도
        "lon": float(lng),  # 경도
        "format": "json",  # 응답 형식 (JSON)
        "addressdetails": 1  # 상세 주소 포함
    }

    try:
        response = requests.get(url, params=params, headers={"User-Agent": "MyGeocoder"})
        response.raise_for_status()  # 요청 실패 시 예외 발생
        data = response.json()

        if "error" in data:  # 오류 메시지 확인
            return "주소를 찾을 수 없음"

        return standardize_address(data.get("address", {}))

    except Exception as e:
        return f"오류 발생: {e}"


if __name__ == "__main__":
    print(reverse_geocode(36.2832456836, 126.9189337881))
