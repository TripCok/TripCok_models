# **TripCok_models**

## Overview

본 github는 `TripCok`의 서비스 중 추천 시스템에 해당하며, `BERT` 계열의 자연어처리모델인 `roBERTa`를 기반으로 한 한국어 모델  `ko-sroberta-multitask`을 사용합니다.
모델에 대한 상세 정보 및 설명은 다음 [huggingface](https://huggingface.co/jhgan/ko-sroberta-multitask) 및 [github](https://github.com/jhgan00/ko-sentence-transformers) 링크에서 확인할 수 있습니다.

본 프로젝트는 해당 모델을 통해 [한국관광공사_국문 관광정보서비스 API](https://www.data.go.kr/data/15101578/openapi.do)에서 받은 여행지 데이터에서 `overview` 칼럼의 개요 텍스트를 embedding으로 변환해, 이를 기반으로 입력값과 데이터셋 내의 여행지들 간의 유사도를 판별, 이를 기반으로 입력값과 가장 적합하다 판별되는 여행지를 추천하는 방식으로 동작합니다.


## **환경 (Environment)**

- **Python**: >= 3.11


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
