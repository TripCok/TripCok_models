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

## 프로젝트 관련 주요 참고사항

- 본 프로젝트는 TripCok-Server와 통합적으로 동작합니다.