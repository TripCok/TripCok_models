# RoundRobin 클래스 정의
class RoundRobin:
    def __init__(self, items):
        self.items = items
        self.index = -1

    def next(self):
        """다음 아이템 반환"""
        if not self.items:
            return None
        self.index = (self.index + 1) % len(self.items)
        return self.items[self.index]

# 환경 변수에서 API 키를 가져오는 함수
def get_keys():
    import os
    API_KEY1 = os.getenv("TOURISM_API1")
    API_KEY2 = os.getenv("TOURISM_API2")
    API_KEY3 = os.getenv("TOURISM_API3")
    API_KEY4 = os.getenv("TOURISM_API4")
    return [key for key in [API_KEY1, API_KEY2, API_KEY3, API_KEY4] if key]  # None 필터링

# get_keys() 함수 호출로 반환 값을 전달
round_robin = RoundRobin(get_keys())

# API 키 순환 출력 테스트
for _ in range(10):
    print(round_robin.next())
