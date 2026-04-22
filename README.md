# 한국어 악성 텍스트 우회 공격 탐지 연구

> **연구 제목**: 한국어 문자 특성 기반 악성 텍스트 우회 공격 분류 및 AI 탐지 모델 취약성 분석  
> **연구자**: 이기찬, 김종헌  
> **지도교사**: 송현지

---

## 📁 프로젝트 폴더 구조 설명

```
korean-adversarial-nlp/
│
├── 📂 data/                    ← 모든 데이터 파일이 들어가는 곳
│   ├── raw/                    ← 원본 데이터 (손대지 말 것!)
│   ├── processed/              ← 정제된 데이터
│   └── augmented/              ← 공격 유형이 적용된 변형 데이터
│
├── 📂 src/                     ← 실제 코드가 들어가는 곳 (핵심!)
│   ├── attacks/                ← 9가지 공격 유형 코드
│   ├── models/                 ← AI 모델 불러오기 및 학습 코드
│   ├── evaluation/             ← 성능 측정 코드
│   └── utils/                  ← 자주 쓰는 도구 모음
│
├── 📂 configs/                 ← 실험 설정값 저장 (학습률, 배치 크기 등)
│
├── 📂 notebooks/               ← 실험 노트 (Jupyter Notebook, 결과 확인용)
│
├── 📂 results/                 ← 실험 결과 저장
│   ├── logs/                   ← 학습 과정 기록
│   ├── metrics/                ← F1 점수 등 성능 수치
│   └── figures/                ← 그래프, 시각화 이미지
│
├── 📂 scripts/                 ← 실험을 "한 번에 실행"하는 스크립트
│
├── README.md                   ← 지금 읽고 있는 이 파일
├── requirements.txt            ← 필요한 파이썬 패키지 목록
└── .gitignore                  ← Git에 올리지 말아야 할 파일 목록
```

---

## 🚀 실험 실행 방법

### 1단계: 환경 설치 (처음 한 번만)
```bash
pip install -r requirements.txt
```

### 2단계: 데이터 준비
```bash
python scripts/prepare_data.py
```
→ `data/raw/` 안의 K-HATERS, KOLD 데이터를 정제해서 `data/processed/`에 저장

### 3단계: 공격 데이터셋 생성
```bash
python scripts/run_attacks.py --attack all --intensity 0.2
```
→ `src/attacks/`의 코드를 이용해 변형 데이터를 `data/augmented/`에 저장  
→ `--attack` 옵션: `jamo`, `phoneme`, `visual`, `coda`, `liaison`, `spacing`, `roman`, `emoji`, `combined`, `all`  
→ `--intensity` 옵션: 변형 강도 (`0.1`, `0.2`, `0.3`)

### 4단계: 소형 모델 학습 및 평가
```bash
python scripts/run_finetune.py --model klue-bert --config configs/finetune.yaml
```

### 5단계: LLM 평가 (API 사용)
```bash
python scripts/run_llm_eval.py --model gpt-4o --config configs/llm.yaml
```

### 6단계: 결과 정리
```bash
python scripts/summarize_results.py
```
→ `results/metrics/`에 CSV, `results/figures/`에 그래프 저장

---

## 🔬 9가지 공격 유형 요약

| 번호 | 공격 이름 | 예시 | 코드 파일 |
|------|-----------|------|-----------|
| 1 | 자모 분리 | 바보 → ㅂㅏㅂㅗ | `src/attacks/jamo_split.py` |
| 2 | 음소 치환 | 바보 → 바뵤 | `src/attacks/phoneme_sub.py` |
| 3 | 시각적 유사 문자 대체 | 바보 → 바B0 | `src/attacks/visual_sub.py` |
| 4 | 받침 탈락/삽입 | 바보 → 박보 | `src/attacks/coda_manip.py` |
| 5 | 연음 역이용 | 먹어 → 머거 | `src/attacks/liaison.py` |
| 6 | 띄어쓰기 조작 | 나쁜놈 → 나 쁜 놈 | `src/attacks/spacing.py` |
| 7 | 로마자 혼용 | 바보 → babo | `src/attacks/romanize.py` |
| 8 | 이모지/특수문자 삽입 | 바보 → 바★보 | `src/attacks/emoji_insert.py` |
| 9 | 복합 공격 | 위 중 2개 이상 조합 | `src/attacks/combined.py` |

---

## 📊 실험 대상 모델

| 분류 | 모델 이름 | 비용 | 학습 방식 |
|------|-----------|------|-----------|
| 소형 모델 | KLUE-BERT | 무료 | Fine-tuning |
| 소형 모델 | KLUE-RoBERTa | 무료 | Fine-tuning |
| 소형 모델 | KCBERT | 무료 | Fine-tuning |
| 대형 모델 | GPT-4o | 유료 (API) | Zero-shot |
| 대형 모델 | HyperCLOVA X | 유료 (API) | Zero-shot |

> **Fine-tuning**: 모델을 우리 데이터로 추가 학습시키는 것  
> **Zero-shot**: 별도 학습 없이 바로 질문하는 것

---

## 📚 참고 논문

| 논문 | 내용 |
|------|------|
| Bitton et al. (2022) - ATN | 영어 텍스트 공격 유형 정의 |
| Yu et al. (2024) - NAACL | 한국어 3가지 공격 유형 + 방어 |
| Kim et al. (2025) - PHISH/MESH | 한국어 음소 치환 공격 + 방어 |
| Park et al. (2023) - K-HATERS | 한국어 혐오표현 데이터셋 |
| Jeong et al. (2022) - KOLD | 한국어 공격표현 데이터셋 |
