# **TripCok_models**

### **K-Means Clustering & K-Nearest Neighbor**

이 프로젝트는 K-Means 클러스터링 및 K-Nearest Neighbor(KNN) 알고리즘을 사용한 모델 적용 및 테스트를 다룹니다.

---

## **환경 (Environment)**

- **Python**: >= 3.11

---

## **설치 가이드 (Setup Guide)**

### **권장 설치 과정**

```bash
# 1. 가상환경 생성
$ pdm venv create

# 2. 가상환경 활성화
$ source .venv/bin/activate

# 3. 의존성 설치
$ pdm install

# 4. 패키지 설치
$ pip install .
```

## Auto Load Category (ALC)

### ALC 실행 방법

1. 위 설치 과정을 완료합니다.
2. TripCok-Server를 구동합니다.
3. 다음 명령어를 입력하여 실행합니다:

```bash
$ alc
```

### ALC 주요 기능

- CSV 데이터를 기반으로 카테고리를 자동 생성합니다.
- 서버와 통신하여 부모-자식 관계의 카테고리를 구성합니다.
- 실행 시 관리자 ID를 입력받아 프로세스를 시작합니다.

---

## Auto Load Places (ALP)

### ALP 실행 방법

1. 위 설치 과정을 완료합니다.
2. TripCok-Server를 구동합니다.
3. 다음 명령어를 입력하여 실행합니다:

```bash
$ python -m tripcok_models.worker.ALP
```

### ALP 주요 기능

- CSV 파일에서 여행지 데이터를 읽어와 데이터베이스에 저장합니다.
- 다음과 같은 작업 흐름으로 동작합니다:
    1. CSV 파일 읽기:
        - 지정된 디렉토리에서 batch_*.csv 형식의 파일을 검색합니다.
    2. 좌표 기반 주소 변환:
        - reverse_geocode를 사용하여 위도와 경도로 주소를 추출합니다.
    3. 여행지 데이터 저장:
        - 여행지 정보(title, description, longitude, latitude 등)를 place 테이블에 저장합니다.
    4. 카테고리 정보 처리:
        - get_categories를 사용하여 카테고리 계층을 생성 및 저장합니다.
    5. 이미지 정보 저장:
        - 이미지 URL 정보를 place_image 테이블에 저장합니다.
    6. 진행 상태 관리:
        - 작업 도중 중단된 경우, 중단된 위치부터 다시 시작할 수 있습니다.

---

### ALP 주요 설정 및 입력

1. 실행 시 입력 정보:

- DB IP: 데이터베이스 서버 IP. (기본값: 127.0.0.1)
- DB Port: 데이터베이스 포트 번호. (기본값: 3306)
- DB Username: 데이터베이스 사용자 이름. (기본값: root)
- DB Password: 데이터베이스 비밀번호. (기본값: root)
- DB Name: 데이터베이스 이름. (기본값: database)

2. CSV 파일 경로:

- 기본 경로는 ../csv_maker/batch이며, 파일 이름은 batch_0.csv, batch_1.csv 등의 형식을 따라야 합니다.

3. 진행 상태 관리:

- 작업 진행 상태는 progress.json 파일에 저장됩니다.
- 예:
  ```json
    {
        "batch_0.csv": -1,
        "batch_1.csv": 5
    }
    ```
    - -1: 파일 처리가 완료됨.
    - 5: 현재 파일의 6번째 행까지 처리됨 (0부터 시작).
  
###  ALP 주요 함수

1. read_csv
   - 지정된 경로에서 처리할 CSV 파일을 검색 및 정렬합니다.
2. process_csv
   - CSV 파일의 각 행을 읽고 다음 작업을 수행:
     - 좌표 기반 주소 변환.
     - 여행지 정보를 place 테이블에 저장.
     - 카테고리 계층을 생성 및 저장.
     - 이미지 URL 정보를 저장.
3. save_progress / load_progress
   - 작업 진행 상태를 저장 및 복원합니다.
4. ask_user_choice
   - 작업을 처음부터 다시 시작할지, 이전 진행 상태를 이어서 실행할지 결정합니다.

---

### 프로젝트 관련 주요 참고사항
- 본 프로젝트는 TripCok-Server와 통합적으로 동작합니다.
- ALP는 대규모 데이터를 처리할 수 있도록 설계되었으며, 작업 도중 중단된 경우에도 재시작 기능을 제공합니다.
- ALC 및 ALP는 각각 카테고리와 여행지 데이터를 처리하여 서버와 데이터베이스를 동기화하는 핵심 도구입니다.
