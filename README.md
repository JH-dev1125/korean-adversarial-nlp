# 한국어 악성 텍스트 우회 공격 탐지 연구

> 연구 제목: 한국어 문자 특성 기반 악성 텍스트 우회 공격 분류 및 AI 탐지 모델 취약성 분석  
> 연구자: 이기찬, 김종헌  
> 지도교사: 송현지

이 프로젝트는 한국어 혐오/공격 표현 탐지 모델이 문자 변형 공격에 얼마나 취약한지 분석하기 위한 연구 코드입니다.  
K-HATERS, KOLD, K-MHaS 데이터를 하나의 이진 분류 데이터셋으로 통합하고, 한국어 문자 특성을 이용한 공격 데이터를 생성한 뒤, KLUE-BERT, KLUE-RoBERTa, KCBERT를 fine-tuning하여 원본 테스트와 변형 테스트 성능을 비교합니다.

현재 핵심 실험은 소형 언어 모델 3개를 학습하고, 원본 test.csv와 공격 적용 test_*.csv의 F1 차이를 비교하는 것입니다. LLM 평가는 이후 확장 단계입니다.

---

## 프로젝트 구조

```text
korean-adversarial-nlp/
│
├── configs/
│   └── finetune.yaml                 # 소형 모델 학습 설정
│
├── data/
│   ├── raw/                          # 원본 데이터셋 저장 위치
│   ├── processed/                    # 전처리된 train/val/test.csv
│   └── augmented/                    # 공격 적용 test_*.csv
│
├── results/
│   ├── logs/                         # 학습 로그
│   ├── metrics/                      # 평가 결과 CSV
│   ├── figures/                      # 논문용 그래프
│   └── saved_models/                 # fine-tuned 모델 저장 위치
│
├── src/
│   ├── make_datasets/
│   │   ├── download.py               # K-HATERS, KOLD, K-MHaS 다운로드
│   │   └── preprocess.py             # 라벨 통합 및 train/val/test 분할
│   │
│   ├── attacks/
│   │   ├── base_attack.py            # 공격 공통 부모 클래스
│   │   ├── hangul_utils.py           # 한글 음절 분해/조합 유틸
│   │   ├── jamo_split.py             # 자모 분리
│   │   ├── phoneme_sub.py            # 음소 치환
│   │   ├── visual_sub.py             # 시각적 유사 문자
│   │   ├── coda_manip.py             # 받침 탈락/삽입
│   │   ├── liaison.py                # 연음 역이용
│   │   ├── spacing.py                # 띄어쓰기 조작
│   │   ├── romanize.py               # 로마자 혼용
│   │   ├── emoji_insert.py           # 특수문자 삽입
│   │   ├── korean_to_english_typing.py # 영타 변환
│   │   ├── compound_attack.py        # 2개 이상 공격 조합
│   │   └── run_all_attacks.py        # 모든 공격 데이터 생성
│   │
│   ├── models/
│   │   ├── model_utils.py            # 공통 설정, Dataset, 지표 함수
│   │   ├── model_train.py            # 소형 모델 fine-tuning
│   │   └── model_evaluate.py         # 원본/공격 테스트 평가
│   │
│   └── evaluation/
│       └── visualize.py              # 결과 CSV를 논문용 그래프로 변환
│
├── requirements.txt
├── PROJECT_HANDOFF.md
└── README.md
```

`notebooks/`는 제거했습니다. 실험 로직은 노트북이 아니라 `src/` 안의 Python 모듈로 실행합니다.

---

## 실험 흐름

전체 실험은 아래 순서로 진행됩니다.

```text
1. 원본 데이터 다운로드
2. 전처리 및 train/val/test 분할
3. 공격 데이터셋 생성
4. 소형 모델 fine-tuning
5. 원본 test와 공격 test 평가
6. CSV 결과를 그래프로 시각화
```

---

## 설치

로컬 또는 Kaggle에서 프로젝트 루트로 이동한 뒤 설치합니다.

```bash
pip install -r requirements.txt
```

`requirements.txt`는 현재 코드에서 실제로 import되는 패키지 중심으로 구성되어 있습니다.

핵심 버전:

```text
transformers==4.44.2
peft==0.13.2
accelerate==0.34.2
datasets>=2.19.0,<3.0.0
torch>=2.2.0,<3.0.0
```

Hugging Face `Trainer`는 내부적으로 `accelerate`를 사용합니다. 또한 Kaggle 환경에 이미 설치된 `peft`와 낮은 `transformers` 버전이 맞지 않으면 `EncoderDecoderCache` import 오류가 날 수 있으므로, `requirements.txt`에서 `transformers`, `peft`, `accelerate`를 함께 맞춰 둡니다.

현재 공격 코드는 자체 한글 유틸을 사용하므로 `jamo`, `g2pk`, `konlpy`는 기본 설치에서 제외했습니다. LLM 평가 코드를 추가하기 전까지 `openai`도 기본 설치에 포함하지 않습니다.

---

## 실행 방법

모든 명령은 프로젝트 최상위 폴더에서 실행합니다.

### 1. 데이터 다운로드

```bash
python -m src.make_datasets.download
```

출력:

```text
data/raw/khaters/
data/raw/kold/
data/raw/kmhas/
```

### 2. 전처리

```bash
python -m src.make_datasets.preprocess
```

출력:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

라벨은 다음처럼 통일합니다.

```text
0 = 일반 표현
1 = 혐오/공격 표현
```

### 3. 공격 데이터셋 생성

기본값은 기존 실험 설계처럼 `label=1`, 즉 혐오/공격 표현 문장에만 공격을 적용합니다. `label=0` 일반 문장은 변형하지 않고 복사합니다.

```bash
python -m src.attacks.run_all_attacks
```

`label=0` 일반 문장에도 공격을 적용하려면 `--attack_label0`을 추가합니다.

```bash
python -m src.attacks.run_all_attacks --attack_label0
```

출력:

```text
data/augmented/test_jamo_0.1.csv
data/augmented/test_jamo_0.2.csv
data/augmented/test_jamo_0.3.csv
...
data/augmented/test_compound_0.1.csv
data/augmented/test_compound_0.2.csv
data/augmented/test_compound_0.3.csv
```

`--attack_label0`을 사용하면 기존 파일을 덮어쓰지 않도록 파일명에 `all_labels`가 붙습니다.

```text
data/augmented/test_jamo_all_labels_0.1.csv
data/augmented/test_jamo_all_labels_0.2.csv
data/augmented/test_jamo_all_labels_0.3.csv
...
data/augmented/test_compound_all_labels_0.3.csv
```

기본 설정:

```text
공격 강도: 0.1, 0.2, 0.3
variant_id: 원문 하나당 5개
기본 모드: label=1 공격 적용, label=0 원문 복사
--attack_label0 모드: label=1과 label=0 모두 공격 적용
```

### 4. 모델 학습

모델 하나만 학습:

```bash
python -m src.models.model_train --model kcbert
```

다른 모델:

```bash
python -m src.models.model_train --model klue-bert
python -m src.models.model_train --model klue-roberta
```

세 모델을 한 번에 학습:

```bash
python -m src.models.model_train --model all
```

출력:

```text
results/saved_models/kcbert/best/
results/saved_models/klue-bert/best/
results/saved_models/klue-roberta/best/
```

학습 설정은 `configs/finetune.yaml`에서 관리합니다.

### 5. 모델 평가

기본 평가는 공격 데이터에서 `variant_id=1`만 사용합니다. 평가 코드는 공격 CSV의 `attack_label0` 컬럼 또는 파일명 `all_labels`를 보고 두 평가 방식을 자동으로 구분합니다.

```text
label1_only 파일:
  label=0은 원본 test.csv 예측 재사용
  label=1은 공격된 문장 예측 사용

all_labels 파일:
  label=0과 label=1 모두 공격된 CSV 전체를 새로 예측
```

```bash
python -m src.models.model_evaluate --model kcbert
```

모든 variant를 평가하려면 `--all_variants`를 추가합니다.

```bash
python -m src.models.model_evaluate --model kcbert --all_variants
```

세 모델 평가:

```bash
python -m src.models.model_evaluate --model all
```

출력:

```text
results/metrics/kcbert_results.csv
results/metrics/klue-bert_results.csv
results/metrics/klue-roberta_results.csv
```

결과 CSV에는 `attack_scope` 컬럼이 추가됩니다.

```text
attack_scope=label1_only  # label=1만 공격한 기존 방식
attack_scope=all_labels   # label=0과 label=1 모두 공격한 방식
```

중간 저장 파일:

```text
results/metrics/{model}_results_partial.csv
```

Kaggle 세션이 중간에 끊겼을 때 어디까지 평가됐는지 확인할 수 있습니다.

### 6. 시각화

```bash
python -m src.evaluation.visualize
```

주의: `label1_only` 결과와 `all_labels` 결과를 같은 `results/metrics/`에 동시에 두면, 현재 시각화 코드는 같은 공격 유형의 결과를 평균낼 수 있습니다. 논문용 그래프를 만들 때는 한 모드의 결과만 남겨두거나, 결과 폴더를 나누어 시각화하는 것을 권장합니다.

출력:

```text
results/figures/fig1_delta_f1_by_attack.png
results/figures/fig1_delta_f1_by_attack.pdf
results/figures/fig2_delta_f1_heatmap.png
results/figures/fig2_delta_f1_heatmap.pdf
results/figures/fig3_delta_f1_by_intensity.png
results/figures/fig3_delta_f1_by_intensity.pdf
results/figures/fig4_fn_rate_by_attack.png
results/figures/fig4_fn_rate_by_attack.pdf
results/metrics/summary_delta_f1.csv
results/metrics/summary_fn_rate.csv
```

---

## Kaggle에서 실행하기

Kaggle 새 Notebook을 만들고, Settings에서 다음을 켭니다.

```text
Internet: On
Accelerator: GPU T4 x2 또는 GPU
```

그 다음 셀을 순서대로 실행합니다.

### 셀 1. GitHub에서 프로젝트 가져오기

```python
%cd /kaggle/working
!git clone https://github.com/JH-dev1125/korean-adversarial-nlp.git
%cd /kaggle/working/korean-adversarial-nlp
```

### 셀 2. 패키지 설치

```python
!python -m pip install -q -r requirements.txt
```

만약 이 셀을 실행하기 전에 이미 `transformers`, `peft`, `torch`를 import했다면, Kaggle 메뉴에서 `Run > Restart & clear cell outputs`를 한 번 누른 뒤 셀 1부터 다시 실행하는 것이 안전합니다.

### 셀 3. 패키지 import 및 GPU 확인

```python
import torch
import transformers
import datasets
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns
import yaml
import accelerate
import peft

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EncoderDecoderCache,
)

print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
print("datasets:", datasets.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("GPU 사용 가능:", torch.cuda.is_available())
!nvidia-smi
```

위 셀이 오류 없이 끝나고 `GPU 사용 가능: True`가 나오면 학습을 진행할 수 있습니다.

### 셀 4. 데이터 준비

```python
!python -m src.make_datasets.download
!python -m src.make_datasets.preprocess
```

### 셀 5. 공격 데이터 생성

기존 방식, 즉 혐오 표현 `label=1`만 공격:

```python
!python -m src.attacks.run_all_attacks
```

일반 표현 `label=0`도 함께 공격:

```python
!python -m src.attacks.run_all_attacks --attack_label0
```

두 명령을 모두 실행하면 `data/augmented/`에 두 종류의 공격 파일이 함께 생깁니다.

### 셀 6. 모델 학습

KCBERT 예시:

```python
!python -m src.models.model_train --model kcbert
```

KLUE-BERT:

```python
!python -m src.models.model_train --model klue-bert
```

KLUE-RoBERTa:

```python
!python -m src.models.model_train --model klue-roberta
```

### 셀 7. 모델 평가

```python
!python -m src.models.model_evaluate --model kcbert
```

모든 variant 평가:

```python
!python -m src.models.model_evaluate --model kcbert --all_variants
```

### 셀 8. 그래프 생성 및 압축

```python
!python -m src.evaluation.visualize
!zip -r results_kcbert.zip results/metrics results/figures results/saved_models/kcbert
```

Kaggle에서 웹페이지를 닫아도 계속 실행되게 하려면 `Save Version` 또는 `Save & Run All`을 사용합니다. 실행이 끝나면 Output에서 결과 파일을 확인할 수 있습니다.

모델 파일이 너무 커서 직접 다운로드하기 어렵다면, Kaggle Output에서 Dataset을 만들어 공유하는 방식을 권장합니다.

---

## 공격 유형

| 번호 | 공격 유형 | 예시 | 파일 |
|---|---|---|---|
| 1 | 자모 분리 | 바보 -> ㅂㅏㅂㅗ | `src/attacks/jamo_split.py` |
| 2 | 음소 치환 | 김 -> 킴 | `src/attacks/phoneme_sub.py` |
| 3 | 시각적 유사 문자 | ㅇ -> 0 | `src/attacks/visual_sub.py` |
| 4 | 받침 탈락/삽입 | 밥 -> 바 / 바 -> 박 | `src/attacks/coda_manip.py` |
| 5 | 연음 역이용 | 먹어 -> 머거 | `src/attacks/liaison.py` |
| 6 | 띄어쓰기 조작 | 나쁜놈 -> 나 쁜 놈 | `src/attacks/spacing.py` |
| 7 | 로마자 혼용 | 바보 -> babo | `src/attacks/romanize.py` |
| 8 | 특수문자 삽입 | 바보 -> 바★보 | `src/attacks/emoji_insert.py` |
| 9 | 영타 변환 | 시발 -> tlqkf | `src/attacks/korean_to_english_typing.py` |
| 10 | 복합 공격 | 위 공격 중 2개 이상 조합 | `src/attacks/compound_attack.py` |

복합 공격은 각 문장마다 2개 또는 3개의 단일 공격을 무작위로 선택해 순차 적용합니다. 결과 CSV에는 `component_attacks`, `compound_metadata` 컬럼이 추가되어 어떤 공격 조합이 쓰였는지 기록됩니다.

---

## 사용 모델

| 명령어 이름 | Hugging Face 모델 | 설명 |
|---|---|---|
| `klue-bert` | `klue/bert-base` | KLUE 기반 한국어 BERT |
| `klue-roberta` | `klue/roberta-base` | KLUE 기반 한국어 RoBERTa |
| `kcbert` | `beomi/kcbert-base` | 한국어 댓글 데이터 기반 BERT |

모델 이름 매핑은 `src/models/model_utils.py`의 `MODEL_MAP`에서 관리합니다.

---

## 결과 CSV 해석

평가 결과 CSV에는 다음 컬럼이 포함됩니다.

| 컬럼 | 의미 |
|---|---|
| `model` | 평가한 모델 이름 |
| `attack_type` | 공격 유형. `none`은 원본 테스트 |
| `intensity` | 공격 강도 |
| `f1` | 해당 조건에서의 macro F1 |
| `baseline_f1` | 원본 테스트 F1 |
| `delta_f1` | `baseline_f1 - attacked_f1` |
| `accuracy` | 정확도 |
| `tn` | 일반 문장을 일반으로 맞힌 수 |
| `fp` | 일반 문장을 혐오로 잘못 예측한 수 |
| `fn` | 혐오 문장을 일반으로 놓친 수 |
| `tp` | 혐오 문장을 혐오로 맞힌 수 |
| `fp_rate` | `FP / (TN + FP)` |
| `fn_rate` | `FN / (TP + FN)` |

`delta_f1` 해석:

```text
delta_f1 > 0  : 공격 후 F1이 감소함. 공격이 모델 성능을 떨어뜨림.
delta_f1 = 0  : 공격 전후 F1 변화 없음.
delta_f1 < 0  : 공격 후 F1이 오히려 증가함.
```

따라서 공격의 강도를 볼 때는 `delta_f1`이 클수록 공격이 더 강하게 작용했다고 볼 수 있습니다. 반대로 모델의 견고성을 볼 때는 `delta_f1`이 작을수록 더 안정적인 모델입니다.

---

## 현재 실험 해석 시 주의점

현재 공격 생성 방식은 문장 안의 변형 가능한 위치를 무작위로 고릅니다. 즉, 실제 욕설이나 혐오 표현이 등장하는 핵심 구간만 골라 변형하는 방식은 아직 아닙니다.

이 때문에 일부 모델에서는 공격 후 F1이 오히려 올라갈 수 있습니다. 변형 문자가 혐오 표현을 숨기기보다, 모델에게 "비정상적인 악성 댓글 스타일"이라는 추가 단서처럼 작용할 수 있기 때문입니다.

향후 개선 방향은 다음과 같습니다.

```text
1. 욕설/혐오 표현 사전 구축
2. 문장 내 공격 대상 span 탐지
3. 해당 span 우선 변형
4. target_found, target_text, fallback_used 같은 메타데이터 저장
```

---

## 참고 문헌

| 번호 | 논문 |
|---|---|
| 1 | Bitton et al., Adversarial Text Normalization, 2022 |
| 2 | Yu et al., Don't be a Fool: Pooling Strategies in Offensive Language Detection, 2024 |
| 3 | Kim et al., PHISH in MESH: Korean Adversarial Phonetic Substitution and Phonetic-Semantic Feature Integration Defense, 2025 |
| 4 | Park et al., K-HATERS: A Hate Speech Detection Corpus in Korean with Target-Specific Ratings, 2023 |
| 5 | Jeong et al., KOLD: Korean Offensive Language Dataset, 2022 |
| 6 | Lee, J., KcBERT: Korean Comments BERT, 2020 |
| 7 | Park et al., KLUE: Korean Language Understanding Evaluation, 2021 |
