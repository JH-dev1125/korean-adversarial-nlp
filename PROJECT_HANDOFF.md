# 프로젝트 인수인계 문서

이 문서는 한국어 악성/혐오 텍스트 우회 공격 탐지 연구 프로젝트를 Claude Code 또는 다른 개발 도구가 이어받을 수 있도록 정리한 작업 기록이다.

## 1. 연구 목적

이 프로젝트는 한국어 혐오 표현 탐지 모델이 글자 변형 공격에 얼마나 취약한지 분석한다.

핵심 실험은 다음과 같다.

1. 원본 한국어 혐오 표현 데이터셋으로 소형 언어 모델을 fine-tuning한다.
2. 원본 test 데이터로 기본 성능을 측정한다.
3. 자모 분리, 음소 치환, 시각적 유사 문자, 받침 조작, 연음, 띄어쓰기, 로마자 혼용, 특수문자 삽입, 복합 공격을 적용한 변형 test 데이터로 다시 평가한다.
4. 원본 F1과 공격 후 F1을 비교하여 공격 유형별 취약성을 분석한다.

현재 우선순위는 LLM 평가가 아니라, 소형 모델 3개를 안정적으로 학습/평가하는 것이다.

사용 모델:

- `klue-bert`
- `klue-roberta`
- `kcbert`

## 2. 현재 코드 구조

중요한 파일은 다음과 같다.

```text
src/models/model_utils.py
src/models/train_small_model.py
src/models/evaluate_small_model.py
src/evaluation/visualize.py
src/attacks/run_all_attacks.py
src/attacks/compound_attack.py
```

현재 모델 실행 방식은 `small_model.py` 하나를 직접 실행하는 방식이 아니라, 모듈을 나누어 실행하는 방식이다.

```bash
python -m src.models.train_small_model --model kcbert
python -m src.models.evaluate_small_model --model kcbert
python -m src.evaluation.visualize
```

모델 전체를 실행하려면 다음처럼 한다.

```bash
python -m src.models.train_small_model --model all
python -m src.models.evaluate_small_model --model all
```

## 3. 중요 변경 이력

### small_model.py 구조 변경

초기에는 `src/models/small_model.py`에서 학습과 평가를 모두 처리했다.
이후 동료가 코드를 나누면서 현재는 다음 구조가 되었다.

- `model_utils.py`: 공통 상수, 데이터셋, 모델 로딩 유틸
- `train_small_model.py`: fine-tuning과 모델 저장
- `evaluate_small_model.py`: 원본 test와 변형 test 평가

따라서 Kaggle에서는 절대 아래처럼 실행하면 안 된다.

```bash
python src/models/train_small_model.py
```

이 방식은 `ModuleNotFoundError: No module named 'src.models'`를 일으킬 수 있다.

반드시 프로젝트 루트에서 아래처럼 실행해야 한다.

```bash
python -m src.models.train_small_model --model kcbert
```

### 복합 공격 추가

최근 `src/attacks/compound_attack.py`를 추가했다.

복합 공격은 매 변형마다 단일 공격 중 2개 또는 3개를 무작위로 선택하고, 전체 공격 강도를 선택된 공격들에 비율로 나누어 순차 적용한다.

생성 파일 예:

```text
data/augmented/test_compound_0.1.csv
data/augmented/test_compound_0.2.csv
data/augmented/test_compound_0.3.csv
```

복합 공격 CSV에는 추가 컬럼이 있다.

```text
component_attacks
compound_metadata
```

`compound_metadata`에는 공격 순서, 공격 유형, 적용 비율, component intensity, 실제 변경 여부가 JSON으로 저장된다.

## 4. Kaggle에서 권장 실행 순서

Kaggle Notebook에서는 셀을 다음 순서로 구성하는 것을 권장한다.

### 셀 1: 프로젝트 가져오기

```python
%cd /kaggle/working
!git clone https://github.com/JH-dev1125/korean-adversarial-nlp.git
%cd /kaggle/working/korean-adversarial-nlp
```

### 셀 2: 의존성 설치

```python
!pip uninstall -y -q peft
!pip install -q transformers==4.40.0 datasets==2.19.0 pandas scikit-learn matplotlib seaborn 'tqdm>=4.66.3'
```

`WARNING: Skipping peft as it is not installed`는 문제 없다.

Kaggle에서 dependency conflict warning이 많이 뜰 수 있다. 대부분은 Kaggle 기본 패키지와 버전이 맞지 않는다는 경고이며, `transformers`, `torch`, `pandas`, `sklearn` import가 정상이고 학습이 시작되면 일단 진행해도 된다.

### 셀 3: GPU 확인

```python
import torch
import transformers

print("transformers:", transformers.__version__)
print("GPU 사용 가능:", torch.cuda.is_available())

!nvidia-smi
```

### 셀 4: 전처리

```python
!python -m src.utils.preprocess
```

### 셀 5: 공격 데이터 생성

```python
!python -m src.attacks.run_all_attacks
```

### 셀 6: KcBERT 학습

```python
!python -m src.models.train_small_model --model kcbert
```

### 셀 7: KcBERT 평가

```python
!python -m src.models.evaluate_small_model --model kcbert
```

### 셀 8: 시각화

```python
!python -m src.evaluation.visualize
```

### 셀 9: 결과 압축

```python
!zip -r kcbert_results.zip results/metrics results/figures results/saved_models/kcbert
```

## 5. Kaggle 백그라운드 실행 방법

긴 실험은 셀을 직접 하나씩 누르는 것보다 Kaggle의 `Save Version -> Save & Run All`을 사용하는 것이 안전하다.

이 방식은 셀을 위에서 아래로 순서대로 실행한다.

```text
셀 1 -> 셀 2 -> 셀 3 -> ... -> 마지막 셀
```

브라우저를 닫아도 Kaggle 서버가 계속 살아 있으면 실행은 계속된다.
다시 웹에 접속하면 그 순간부터 새로 이어서 하는 것이 아니라, Kaggle 서버에서 이미 진행된 상태를 보여준다.

다만 중간에 실패하면 자동으로 실패 지점부터 재개하지 않는다.
따라서 결과는 반드시 `results/` 아래에 저장하고 마지막에 zip으로 묶어야 한다.

## 6. Kaggle 설정에서 꼭 확인할 것

`Save & Run All`을 누르기 전에 반드시 확인한다.

```text
Internet: On
Accelerator: GPU T4 x2 또는 GPU
```

한 번 실험 중 다음 설정이 보인 적이 있다.

```json
"accelerator": "none",
"isInternetEnabled": false,
"isGpuEnabled": false
```

이 상태는 KcBERT 실험에 적합하지 않다.

문제:

- GitHub clone이 실패할 수 있다.
- Hugging Face 모델 다운로드가 실패할 수 있다.
- GPU가 없어 학습이 매우 오래 걸릴 수 있다.

따라서 이 상태라면 설정을 바꾼 뒤 새 버전으로 다시 `Save & Run All`을 해야 한다.

## 7. Kaggle에서 겪었던 오류와 해결

### peft와 transformers 충돌

오류:

```text
ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'
```

원인:

Kaggle에 기본 설치된 `peft`가 `transformers==4.40.0`과 맞지 않았다.

해결:

```python
!pip uninstall -y -q peft
!pip install -q transformers==4.40.0 datasets==2.19.0 pandas scikit-learn matplotlib seaborn 'tqdm>=4.66.3'
```

### src.models import 오류

오류:

```text
ModuleNotFoundError: No module named 'src.models'
```

원인:

파일 경로를 직접 실행했다.

잘못된 실행:

```bash
python src/models/train_small_model.py
```

올바른 실행:

```bash
python -m src.models.train_small_model --model kcbert
```

반드시 프로젝트 루트에서 실행해야 한다.

```bash
cd /kaggle/working/korean-adversarial-nlp
```

### CUDA/cuDNN/cuBLAS warning

예:

```text
Unable to register cuFFT factory
Unable to register cuDNN factory
Unable to register cuBLAS factory
```

대부분 Kaggle/TensorFlow/PyTorch 환경에서 흔히 보이는 warning이다.
학습이 계속 진행되면 무시해도 된다.

### visualize.py 결과 파일 없음

오류:

```text
FileNotFoundError: 결과 파일이 없습니다. 먼저 모델 학습을 완료해주세요.
```

원인:

평가 CSV가 생성되기 전에 시각화를 실행했다.

해결 순서:

```bash
python -m src.models.train_small_model --model kcbert
python -m src.models.evaluate_small_model --model kcbert
python -m src.evaluation.visualize
```

## 8. 실험이 제대로 끝났는지 확인하는 파일

KcBERT 실험이 끝났다면 다음 파일/폴더가 있어야 한다.

```text
results/saved_models/kcbert/best
results/metrics/kcbert_results.csv
results/figures/
kcbert_results.zip
```

특히 아래 파일이 가장 중요하다.

```text
results/metrics/kcbert_results.csv
```

이 파일에 다음과 같은 행들이 있어야 한다.

```text
kcbert, none, 0.0, ...
kcbert, jamo, 0.1, ...
kcbert, phoneme, 0.1, ...
kcbert, compound, 0.3, ...
```

## 9. 평가 방식 관련 논의

현재 평가 코드는 원본 test의 label=0 예측과 공격 test의 label=1 예측을 결합하여 공격 후 F1을 계산하는 방식이다.

이 방식은 “혐오 표현만 변형 공격을 받는다”는 실험 설계와 맞다.

다만 `variant_id=1`만 평가했을 때 특정 모델에서 공격 후 F1이 오히려 증가하는 현상이 있었다.
가능한 원인은 다음과 같다.

- 변형이 실제 핵심 욕설이 아니라 주변 글자에 적용되었을 수 있다.
- 랜덤 변형이 일부 문장을 모델에 더 쉽게 만들었을 수 있다.
- variant 하나만 보면 샘플링 편향이 생길 수 있다.
- label=0은 원본 예측을 재사용하고 label=1만 공격하므로, 공격 후 label=1 예측 변화가 전체 F1에 민감하게 반영된다.

더 안정적인 분석을 위해서는 `--all_variants` 평가도 함께 고려할 수 있다.

```bash
python -m src.models.evaluate_small_model --model kcbert --all_variants
```

단, `--all_variants`는 평가 시간이 더 길어진다.

## 10. 공격 위치 랜덤 선택의 한계

현재 대부분의 공격은 문장 내 변형 가능한 위치를 랜덤으로 선택한다.

이 방식의 한계:

- 실제 욕설이나 혐오 표현 핵심 단어가 아닌 부분이 변형될 수 있다.
- 공격이 모델을 속이는 효과보다 단순 노이즈 효과에 가까워질 수 있다.
- 논문에서 “악성 텍스트 우회 공격”이라고 주장하려면 핵심 혐오 표현 부분이 실제로 변형되었는지 메타데이터가 있으면 더 좋다.

개선 방향:

1. 욕설/혐오 표현 lexicon을 만든다.
2. 문장에서 lexicon에 해당하는 span을 찾는다.
3. 가능하면 해당 span 안에서만 변형한다.
4. 찾지 못한 경우에만 기존 랜덤 공격으로 fallback한다.
5. `target_found`, `target_text`, `target_start`, `target_end`, `fallback_used` 같은 메타데이터를 저장한다.

이 개선은 아직 구현하지 않았다.

## 11. 실험 시간을 줄이는 방법

질을 크게 떨어뜨리지 않는 방법:

- 모델 3개를 한 노트북에서 순차 실행하지 말고, 모델별 Kaggle Notebook을 따로 만들어 병렬 실행한다.
- `klue-bert`, `klue-roberta`, `kcbert`를 각각 다른 Notebook에서 실행한다.
- 데이터 전처리와 공격 데이터 생성은 한 번만 만들고 재사용한다.
- main 결과는 `variant_id=1` 기준으로 먼저 만들고, 필요할 때 `--all_variants`를 추가 실험으로 돌린다.
- epoch를 지나치게 줄이지 않는다. 현재 프로젝트에서는 3 epoch가 현실적인 절충안이다.

## 12. Claude Code가 이어받을 때 주의할 점

- `compound_attack.py`는 새로 추가된 파일이므로 커밋 전 상태라면 반드시 git status를 확인한다.
- `run_all_attacks.py`는 복합 공격까지 포함하도록 수정되어 있다.
- `small_model.py`가 없다고 해서 이상한 것이 아니다. 현재는 분리 구조가 맞다.
- Kaggle 실행 명령은 `python -m ...` 형식을 사용해야 한다.
- LLM 평가와 HyperCLOVA X 평가는 아직 우선순위가 아니다.
- 현재 핵심 목표는 소형 모델의 원본 test vs 공격 test 성능 비교이다.

## 13. 현재 남은 주요 작업

1. Kaggle에서 GPU/Internet을 켜고 KcBERT 실험을 완료한다.
2. `results/metrics/kcbert_results.csv`가 정상 생성되는지 확인한다.
3. `compound` 공격 결과가 평가 CSV에 포함되는지 확인한다.
4. 같은 방식으로 `klue-bert`, `klue-roberta`를 실행한다.
5. 세 모델의 CSV를 바탕으로 `visualize.py` 결과물을 논문용으로 검토한다.
6. 필요하면 욕설/핵심 표현 span 기반 targeted attack을 추가 설계한다.
