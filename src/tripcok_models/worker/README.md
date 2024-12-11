# Auto Load Category (ALC)

## **DATA**

```
관광지,A01,A0101,A01010100,자연,자연관광지,국립공원
관광지,A01,A0101,A01010200,자연,자연관광지,도립공원
```

---

## **분석**

- **데이터 구조**:
    - **1번째 열**: 카테고리의 유형
    - **2번, 3번, 4번 열**: 카테고리 고유 번호 (활용하지 않음)
    - **5번, 6번, 7번 열**:
        - 5번 행: 1번째 행 데이터의 **자식 노드**
        - 6번 행: 5번 행 데이터의 **자식 노드**
        - 7번 행: 6번 행 데이터의 **자식 노드**
- **실제 사용할 데이터**:
    - **1번, 5번, 6번, 7번 열**

---

## **Process**

### 사전 처리 사항

- 사용하지 않은 행은 미리 제거
    - 코드
        
        ```python
        import csv
        
        CSV_FILE = "./translate.csv"
        CSV_WRITE = "category.csv"
        with open(CSV_FILE) as csvfile:
            reader = csv.reader(csvfile)
            csv_list = list(reader)
        
        for col in range(len(csv_list)):
            for i in range(4):
                del csv_list[col][0]
        
        csv.writer(open(CSV_WRITE, "w")).writerows(csv_list)
        
        ```
        

### 실제 동작 프로세스

1. **CSV 파일 읽기**:
    - `category.csv` 파일을 읽어서 `csv_list`로 변환합니다.
    - `csv.reader`를 사용해 각 행을 리스트로 처리.
2. **루프 실행**:
    - `csv_list`의 각 행(`row`)을 순회하며 첫 번째, 두 번째, 세 번째 카테고리를 처리.
3. **첫 번째 카테고리 처리**:
    - 첫 번째 카테고리(루트 노드)를 처리하며, `parentId` 없이 API 통신.
    - 카테고리 이름이 이미 존재하는지 확인:
        - **존재하지 않으면**: 새로운 카테고리를 생성.
        - **존재하면**: 기존 카테고리의 ID를 가져옴.
    - **Request**:
        
        ```json
        {
            "depth": 0,
            "categoryName": "첫 번째 카테고리 이름"
        }
        ```
        
    - **POST 요청 (새로운 카테고리 생성)**:
        
        ```json
        {
            "placeName": "첫 번째 카테고리 이름",
            "memberId": 1
        }
        ```
        
    - **Response**:
        
        ```json
        {
            "id": "첫 번째 카테고리 ID",
            "name": "첫 번째 카테고리 이름",
            "children": [],
            "depth": 0
        }
        ```
        
4. **두 번째 카테고리 처리**:
    - 첫 번째 카테고리의 **자식 노드**로 처리하며, `parentId`를 첫 번째 카테고리의 ID로 설정.
    - 카테고리 이름이 이미 존재하는지 확인:
        - **존재하지 않으면**: 새로운 카테고리를 생성.
        - **존재하면**: 기존 카테고리의 ID를 가져옴.
    - **Request**:
        
        ```json
        {
            "depth": 1,
            "categoryName": "두 번째 카테고리 이름",
            "parentId": "첫 번째 카테고리 ID"
        }
        ```
        
    - **POST 요청 (새로운 카테고리 생성)**:
        
        ```json
        {
            "parentId": "첫 번째 카테고리 ID",
            "placeName": "두 번째 카테고리 이름",
            "memberId": 1
        }
        ```
        
    - **Response**:
        
        ```json
        {
            "id": "두 번째 카테고리 ID",
            "name": "두 번째 카테고리 이름",
            "children": [],
            "depth": 1
        }
        ```
        
5. **세 번째 카테고리 처리**:
    - 두 번째 카테고리의 **자식 노드**로 처리하며, `parentId`를 두 번째 카테고리의 ID로 설정.
    - 카테고리 이름이 이미 존재하는지 확인:
        - **존재하지 않으면**: 새로운 카테고리를 생성.
        - **존재하면**: 기존 카테고리의 ID를 가져옴.
    - **Request**:
        
        ```json
        {
            "depth": 2,
            "categoryName": "세 번째 카테고리 이름",
            "parentId": "두 번째 카테고리 ID"
        }
        
        ```
        
    - **POST 요청 (새로운 카테고리 생성)**:
        
        ```json
        {
            "parentId": "두 번째 카테고리 ID",
            "placeName": "세 번째 카테고리 이름",
            "memberId": 1
        }
        ```
        
    - **Response**:
        
        ```json
        {
            "id": "세 번째 카테고리 ID",
            "name": "세 번째 카테고리 이름",
            "children": [],
            "depth": 2
        }
        ```
        
6. **다음 행으로 이동**:
    - 현재 행(row)의 처리가 완료되면 다음 행(row)으로 이동하여 다시 첫 번째 카테고리부터 반복.

---