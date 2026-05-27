# ====================================================
# src/models/small_model.py
# KLUE-BERT, KLUE-RoBERTa, KCBERT 학습 및 평가 코드
#
# 이 파일이 하는 일:
#   1. data/processed/train.csv, val.csv로 모델 학습
#   2. data/processed/test.csv로 기본 성능 측정
#   3. data/augmented/ 변형 데이터로 공격 후 성능 측정
#   4. F1 score, ΔF1 계산 후 results/metrics/에 저장
#
# 실행 방법 (Kaggle/Colab 노트북에서):
#   !python src/models/small_model.py --model klue-bert
#   !python src/models/small_model.py --model klue-roberta
#   !python src/models/small_model.py --model kcbert
#
# 모델 선택:
#   klue-bert    → klue/bert-base
#   klue-roberta → klue/roberta-base
#   kcbert       → beomi/kcbert-base
# ====================================================

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ── 경로 설정 ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "saved_models"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 모델 이름 매핑 ────────────────────────────────────
MODEL_MAP = {
    "klue-bert":    "klue/bert-base",
    "klue-roberta": "klue/roberta-base",
    "kcbert":       "beomi/kcbert-base",
}

# ── 학습 설정 ─────────────────────────────────────────
MAX_LENGTH = 128    # 입력 텍스트 최대 길이
BATCH_SIZE = 32     # 한 번에 처리할 샘플 수
EPOCHS = 5          # 학습 반복 횟수
LEARNING_RATE = 2e-5  # 학습률
SEED = 42


# ====================================================
# 데이터셋 클래스
# ====================================================

class HateSpeechDataset(Dataset):
    """
    파이토치 데이터셋 클래스.
    CSV 파일을 읽어서 모델이 학습할 수 있는 형태로 변환.
    """
    """
    PyTorch Dataset 클래스는 유지하되,
    tokenizer를 __getitem__에서 매번 호출하지 않고
    __init__에서 전체 데이터를 한 번에 토큰화하는 방식.
    """
    def __init__(self, df, tokenizer, max_length=128):
        self.labels = torch.tensor(
            df["label"].tolist(),
            dtype=torch.long
        )

        texts = df["text"].astype(str).tolist()

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
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }

# ====================================================
# 성능 지표 계산 함수
# ====================================================
def compute_metrics(eval_pred):
    """
    학습 중 검증 성능을 계산하는 함수.
    Trainer가 자동으로 호출함.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    return {"f1": f1}


# ====================================================
# 모델 학습 함수
# ====================================================
def train_model(model_key: str):
    """
    모델을 학습하고 저장하는 함수.

    Args:
        model_key: "klue-bert", "klue-roberta", "kcbert" 중 하나
    """
    model_name = MODEL_MAP[model_key]
    print(f"\n{'='*60}")
    print(f"모델 학습 시작: {model_name}")
    print(f"{'='*60}")

    # 토크나이저와 모델 불러오기
    # 토크나이저: 텍스트를 숫자로 변환하는 도구
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # 혐오(1) / 정상(0)
    )

    # 데이터 불러오기
    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "processed" / "val.csv")

    print(f"학습 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(val_df)}개")

    # 데이터셋 생성
    train_dataset = HateSpeechDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = HateSpeechDataset(val_df, tokenizer, MAX_LENGTH)

    # 학습 설정
    save_path = MODELS_DIR / model_key
    training_args = TrainingArguments(
        output_dir=str(save_path),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",  # 매 에포크마다 검증
        save_strategy="epoch",
        load_best_model_at_end=True,  # 가장 좋은 모델 자동 선택
        metric_for_best_model="f1",
        seed=SEED,
        logging_dir=str(RESULTS_DIR / "logs"),
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # GPU 있으면 빠른 학습 모드
    )

    # Trainer: 학습을 자동으로 관리해주는 도구
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 학습 시작!
    trainer.train()

    # 최종 모델 저장
    trainer.save_model(str(save_path / "best"))
    tokenizer.save_pretrained(str(save_path / "best"))
    print(f"\n모델 저장 완료: {save_path / 'best'}")

    return trainer, tokenizer, model


# ====================================================
# 모델 평가 함수
# ====================================================
def evaluate_model(model_key: str, trainer, tokenizer):
    """
    원본 테스트 데이터 + 변형 데이터로 모델 성능 측정.

    Args:
        model_key: 모델 이름
        trainer: 학습된 Trainer 객체
        tokenizer: 토크나이저
    """
    print(f"\n{'='*60}")
    print(f"성능 평가 시작: {model_key}")
    print(f"{'='*60}")

    results = []

    # 1. 원본 테스트 데이터 평가
    test_df = pd.read_csv(DATA_DIR / "processed" / "test.csv")
    test_dataset = HateSpeechDataset(test_df, tokenizer, MAX_LENGTH)
    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=-1)
    f1_original = f1_score(test_df["label"].tolist(), preds, average="macro")

    print(f"\n원본 테스트 F1: {f1_original:.4f}")
    results.append({
        "model": model_key,
        "attack_type": "none",
        "intensity": 0.0,
        "f1": f1_original,
        "delta_f1": 0.0,
    })

    # 2. 변형 데이터 평가
    augmented_dir = DATA_DIR / "augmented"
    for csv_file in sorted(augmented_dir.glob("test_*.csv")):
        aug_df = pd.read_csv(csv_file)

        # label=1인 변형 데이터만 평가 (label=0은 복제본이라 제외)
        # variant_id=1만 사용 (중복 제거)
        aug_df = aug_df[aug_df["variant_id"] == 1].reset_index(drop=True)

        aug_dataset = HateSpeechDataset(aug_df, tokenizer, MAX_LENGTH)
        pred_output = trainer.predict(aug_dataset)
        preds = np.argmax(pred_output.predictions, axis=-1)
        f1_attacked = f1_score(aug_df["label"].tolist(), preds, average="macro")

        # 파일명에서 공격 유형과 강도 추출
        # 예: test_jamo_0.1.csv → attack_type=jamo, intensity=0.1
        parts = csv_file.stem.split("_")  # test_jamo_0.1 → ['test', 'jamo', '0.1']
        attack_type = parts[1]
        intensity = float(parts[2])
        delta_f1 = f1_original - f1_attacked

        print(f"{attack_type} (강도 {intensity}): F1={f1_attacked:.4f}, ΔF1={delta_f1:.4f}")
        results.append({
            "model": model_key,
            "attack_type": attack_type,
            "intensity": intensity,
            "f1": f1_attacked,
            "delta_f1": delta_f1,
        })

    # 결과 저장
    results_df = pd.DataFrame(results)
    save_path = METRICS_DIR / f"{model_key}_results.csv"
    results_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장 완료: {save_path}")

    return results_df


# ====================================================
# 메인 실행
# ====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="klue-bert",
        choices=["klue-bert", "klue-roberta", "kcbert"],
        help="학습할 모델 선택"
    )
    args = parser.parse_args()

    # 학습
    trainer, tokenizer, model = train_model(args.model)

    # 평가
    results_df = evaluate_model(args.model, trainer, tokenizer)

    print(f"\n{'='*60}")
    print(f"✅ 완료! 결과: results/metrics/{args.model}_results.csv")
    print(f"{'='*60}")
