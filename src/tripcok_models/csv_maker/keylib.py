count = -1

def get_keys():
    global count

    import os
    API_KEY1 = os.getenv("TOURISM_API1")
    API_KEY2 = os.getenv("TOURISM_API2")
    API_KEY3 = os.getenv("TOURISM_API3")
    API_KEY4 = os.getenv("TOURISM_API4")
    API_KEY5 = os.getenv("TOURISM_API5")
    API_KEY6 = os.getenv("TOURISM_API6")
    
    key = [API_KEY1, API_KEY2, API_KEY3,API_KEY4, API_KEY5, API_KEY6]
    #key = [API_KEY4, API_KEY5, API_KEY6]
    #key = [API_KEY1, API_KEY2, API_KEY3]
    
    key = [k for k in key if k]

    if not key:  # 키가 없으면 예외 발생
        raise ValueError("환경 변수에서 유효한 API 키를 찾을 수 없습니다!")

    count = (count + 1) % len(key)  # 라운드 로빈 인덱스 업데이트
    return key[count]
