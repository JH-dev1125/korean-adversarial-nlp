"""
src/models/model_utils.py

소형 언어 모델 학습/평가에서 공통으로 쓰는 경로, 데이터셋 클래스,
평가 지표 계산 함수를 모아 둔 파일이다.

이 파일은 직접 실행하지 않는다.
- src/models/model_train.py
- src/models/evaluate_small_model.py
에서 import해서 사용한다.
"""

import random
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
from torch.utils.data import Dataset
from transformers import set_seed as hf_set_seed


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "saved_models"

# --model 인자의 짧은 이름을 Hugging Face 모델 ID로 변환한다.
MODEL_MAP = {
    "klue-bert": "klue/bert-base",
    "klue-roberta": "klue/roberta-base",
    "kcbert": "beomi/kcbert-base",
}


def load_config(config_path=None) -> dict:
    """configs/finetune.yaml을 읽어 딕셔너리로 반환한다."""
    if config_path is None:
        config_path = BASE_DIR / "configs" / "finetune.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class HateSpeechDataset(Dataset):
    """
    pandas DataFrame을 Hugging Face Trainer용 PyTorch Dataset으로 변환한다.

    입력 DataFrame에 text, label 컬럼이 있어야 한다.
    label=0은 정상, label=1은 혐오/공격 표현이다.
    """

    def __init__(self, df, tokenizer, max_length: int = 128):
        # dtype=torch.long은 CrossEntropyLoss가 요구하는 정수형이다.
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)
        texts = df["text"].astype(str).tolist()
        # padding="max_length": 모든 문장을 같은 길이로 맞춰 배치 처리를 쉽게 한다.
        # truncation=True: max_length를 초과하는 문장은 잘라낸다.
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # "labels" 키는 Trainer가 자동으로 인식해 loss 계산에 사용한다.
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def set_random_seed(seed: int = 42):
    """Python, NumPy, PyTorch, Hugging Face의 난수 시드를 고정해 실험 재현성을 높인다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def score_predictions(labels, preds) -> dict:
    """
    정답 라벨과 예측 라벨로 논문/그래프용 지표를 계산한다.

    혼동행렬 용어:
        tn: 정상을 정상으로 맞힘
        fp: 정상을 혐오로 잘못 분류 (오탐)
        fn: 혐오를 정상으로 놓침 (미탐)
        tp: 혐오를 혐오로 맞힘
    """
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)

    # labels=[0, 1]을 명시해야 한쪽 라벨만 있는 경우에도 순서가 보장된다.
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    total_negative = tn + fp
    total_positive = tp + fn

    return {
        "f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fp_rate": fp / total_negative if total_negative else 0.0,
        "fn_rate": fn / total_positive if total_positive else 0.0,
    }


def compute_metrics(eval_pred):
    """
    Trainer가 검증셋 평가 시 호출하는 콜백 함수.

    macro F1은 클래스 불균형이 있을 때 accuracy보다 혐오 탐지 성능 비교에 적합하다.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": score_predictions(labels, preds)["f1"]}


def parse_attack_filename(csv_file: Path) -> tuple[str, float]:
    """
    파일명에서 공격 유형과 강도를 추출한다.

    예:
        test_jamo_0.2.csv -> ("jamo", 0.2)
        test_jamo_all_labels_0.2.csv -> ("jamo", 0.2)

    all_labels는 label=0도 공격했다는 평가 범위 표시이며, 공격 유형 이름에서는 제거한다.
    """
    stem = csv_file.stem.removeprefix("test_")
    attack_type, intensity_str = stem.rsplit("_", 1)
    attack_type = attack_type.removesuffix("_all_labels")
    return attack_type, float(intensity_str)
