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

import argparse
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed as hf_set_seed,
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
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128   # 한 번에 처리할 샘플 수
EPOCHS = 5          # 학습 반복 횟수
LEARNING_RATE = 2e-5  # 학습률
SEED = 42
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


# ====================================================
# 데이터셋 클래스
# ====================================================

class HateSpeechDataset(Dataset):
    """
    파이토치 데이터셋 클래스.
    tokenizer를 __getitem__에서 매번 호출하지 않고 처음 한 번에 토큰화한다.
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
# 재현성 및 평가 보조 함수
# ====================================================
def set_random_seed(seed: int = SEED):
    """
    실험 결과가 최대한 재현되도록 난수 시드를 고정한다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def parse_attack_filename(csv_file: Path) -> tuple[str, float]:
    """
    변형 데이터 파일명에서 공격 유형과 강도를 추출한다.

    예:
        test_jamo_0.1.csv -> ("jamo", 0.1)
        test_engtyping_0.2.csv -> ("engtyping", 0.2)
    """
    if not csv_file.stem.startswith("test_"):
        raise ValueError(f"공격 파일명은 test_로 시작해야 합니다: {csv_file.name}")

    attack_and_intensity = csv_file.stem.removeprefix("test_")
    attack_type, intensity_text = attack_and_intensity.rsplit("_", 1)
    return attack_type, float(intensity_text)


def score_predictions(labels, preds) -> dict:
    """
    정답 라벨과 예측 라벨로 F1, 정확도, 혼동 행렬 값을 계산한다.
    """
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)

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


def predict_labels(trainer, tokenizer, df: pd.DataFrame) -> np.ndarray:
    """
    주어진 데이터프레임에 대한 모델 예측 라벨을 반환한다.
    """
    dataset = HateSpeechDataset(df, tokenizer, MAX_LENGTH)
    pred_output = trainer.predict(dataset)
    return np.argmax(pred_output.predictions, axis=-1)


def predict_and_score(trainer, tokenizer, df: pd.DataFrame) -> dict:
    """
    주어진 데이터프레임을 평가하고 F1, 정확도, 혼동 행렬 값을 반환한다.
    """
    labels = df["label"].astype(int).to_numpy()
    preds = predict_labels(trainer, tokenizer, df)
    return score_predictions(labels, preds)


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
    set_random_seed(SEED)
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
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",  # 매 에포크마다 검증
        save_strategy="epoch",
        load_best_model_at_end=True,  # 가장 좋은 모델 자동 선택
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        logging_dir=str(RESULTS_DIR / "logs"),
        logging_steps=100,
        save_total_limit=1,
        report_to="none",
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
def evaluate_model(model_key: str, trainer, tokenizer, evaluate_all_variants: bool = False):
    """
    원본 테스트 데이터 + 변형 데이터로 모델 성능 측정.

    Args:
        model_key: 모델 이름
        trainer: 학습된 Trainer 객체
        tokenizer: 토크나이저
        evaluate_all_variants: True면 모든 variant_id를 평가하고, False면 variant_id=1만 평가
    """
    print(f"\n{'='*60}")
    print(f"성능 평가 시작: {model_key}")
    print(f"{'='*60}")

    results = []

    # 1. 원본 테스트 데이터 평가
    test_df = pd.read_csv(DATA_DIR / "processed" / "test.csv")
    original_labels = test_df["label"].astype(int).to_numpy()
    original_preds = predict_labels(trainer, tokenizer, test_df)
    original_scores = score_predictions(original_labels, original_preds)
    f1_original = original_scores["f1"]

    print(f"\n원본 테스트 F1: {f1_original:.4f}")
    results.append({
        "model": model_key,
        "attack_type": "none",
        "intensity": 0.0,
        "f1": f1_original,
        "baseline_f1": f1_original,
        "delta_f1": 0.0,
        **original_scores,
    })

    # 정상 텍스트(label=0)는 변형하지 않는 실험 설계이므로 원본 예측 결과를 재사용한다.
    label0_mask = test_df["label"].astype(int) == 0
    label0_labels = original_labels[label0_mask.to_numpy()].tolist()
    label0_preds = original_preds[label0_mask.to_numpy()].tolist()

    # 2. 변형 데이터 평가
    augmented_dir = DATA_DIR / "augmented"
    csv_files = sorted(augmented_dir.glob("test_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"변형 테스트 파일이 없습니다. 먼저 공격 데이터를 생성해주세요: {augmented_dir}"
        )

    for csv_file in csv_files:
        aug_df = pd.read_csv(csv_file)

        # 기본값은 variant_id=1만 평가해 원본 테스트셋과 같은 샘플 수로 비교한다.
        # --all_variants를 쓰면 모든 변형 후보를 평가한다.
        if "variant_id" in aug_df.columns and not evaluate_all_variants:
            aug_df = aug_df[aug_df["variant_id"] == 1].reset_index(drop=True)

        # 혐오 텍스트(label=1)만 실제 공격 변형본으로 평가하고,
        # 정상 텍스트(label=0)는 위에서 저장한 원본 예측 결과를 결합한다.
        aug_label1_df = aug_df[aug_df["label"].astype(int) == 1].reset_index(drop=True)
        label1_labels = aug_label1_df["label"].astype(int).tolist()
        label1_preds = predict_labels(trainer, tokenizer, aug_label1_df).tolist()

        if evaluate_all_variants and "variant_id" in aug_label1_df.columns:
            num_variants = aug_label1_df["variant_id"].nunique()
        else:
            num_variants = 1

        combined_labels = label0_labels * num_variants + label1_labels
        combined_preds = label0_preds * num_variants + label1_preds

        attack_scores = score_predictions(combined_labels, combined_preds)
        f1_attacked = attack_scores["f1"]

        # 파일명에서 공격 유형과 강도 추출
        # 예: test_jamo_0.1.csv → attack_type=jamo, intensity=0.1
        attack_type, intensity = parse_attack_filename(csv_file)
        delta_f1 = f1_original - f1_attacked

        print(f"{attack_type} (강도 {intensity}): F1={f1_attacked:.4f}, ΔF1={delta_f1:.4f}")
        results.append({
            "model": model_key,
            "attack_type": attack_type,
            "intensity": intensity,
            "f1": f1_attacked,
            "baseline_f1": f1_original,
            "delta_f1": delta_f1,
            **attack_scores,
        })

        # Kaggle 런타임이 중간에 끊겨도 일부 결과를 회수할 수 있도록 매 파일마다 중간 저장한다.
        partial_df = pd.DataFrame(results)
        partial_path = METRICS_DIR / f"{model_key}_results_partial.csv"
        partial_df.to_csv(partial_path, index=False, encoding="utf-8-sig")
        print(f"중간 저장 완료: {partial_path}")

    # 결과 저장
    results_df = pd.DataFrame(results)
    save_path = METRICS_DIR / f"{model_key}_results.csv"
    results_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장 완료: {save_path}")

    return results_df


# ====================================================
# 메인 실행
# ====================================================
def run_model(model_key: str, evaluate_all_variants: bool = False):
    """
    한 모델에 대해 학습과 평가를 연속 실행한다.
    """
    trainer, tokenizer, _ = train_model(model_key)
    return evaluate_model(
        model_key,
        trainer,
        tokenizer,
        evaluate_all_variants=evaluate_all_variants,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "klue-bert", "klue-roberta", "kcbert"],
        help="학습할 모델 선택. all이면 3개 모델을 순서대로 실행"
    )
    parser.add_argument(
        "--all_variants",
        action="store_true",
        help="모든 variant_id를 평가. 기본값은 variant_id=1만 평가"
    )
    args = parser.parse_args()

    model_keys = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]
    for model_key in model_keys:
        run_model(model_key, evaluate_all_variants=args.all_variants)

    print(f"\n{'='*60}")
    print("✅ 완료! 결과: results/metrics/")
    print(f"{'='*60}")
