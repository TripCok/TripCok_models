from keylib import get_keys

# ApiManager 사용 예시
if __name__ == "__main__":
    for _ in range(10):  # 10번 키를 순환
        print("사용할 API 키:", get_keys())
