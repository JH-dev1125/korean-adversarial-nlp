# src/models/evaluate_small_model.py

import argparse
import pandas as pd
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.models.model_utils import (
    DATA_DIR,
    RESULTS_DIR,
    METRICS_DIR,
    MODELS_DIR,
    MODEL_MAP,
    EVAL_BATCH_SIZE,
    score_predictions,
    parse_attack_filename,
    predict_labels,
)


def load_trained_model(model_key: str):
    model_path = MODELS_DIR / model_key / "best"

    if not model_path.exists():
        raise FileNotFoundError(
            f"저장된 모델이 없습니다: {model_path}\n"
            f"먼저 train_small_model.py로 학습을 실행하세요."
        )

    print(f"저장된 모델 불러오기: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "tmp_eval" / model_key),
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    return trainer, tokenizer


def evaluate_model(model_key: str, evaluate_all_variants: bool = False):
    trainer, tokenizer = load_trained_model(model_key)

    print(f"\n{'=' * 60}")
    print(f"성능 평가 시작: {model_key}")
    print(f"{'=' * 60}")

    results = []

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

    label0_mask = test_df["label"].astype(int) == 0
    label0_labels = original_labels[label0_mask.to_numpy()].tolist()
    label0_preds = original_preds[label0_mask.to_numpy()].tolist()

    augmented_dir = DATA_DIR / "augmented"
    csv_files = sorted(augmented_dir.glob("test_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"변형 테스트 파일이 없습니다: {augmented_dir}"
        )

    for csv_file in csv_files:
        aug_df = pd.read_csv(csv_file)

        if "variant_id" in aug_df.columns and not evaluate_all_variants:
            aug_df = aug_df[aug_df["variant_id"] == 1].reset_index(drop=True)

        aug_label1_df = aug_df[
            aug_df["label"].astype(int) == 1
        ].reset_index(drop=True)

        label1_labels = aug_label1_df["label"].astype(int).tolist()
        label1_preds = predict_labels(
            trainer,
            tokenizer,
            aug_label1_df
        ).tolist()

        if evaluate_all_variants and "variant_id" in aug_label1_df.columns:
            num_variants = aug_label1_df["variant_id"].nunique()
        else:
            num_variants = 1

        combined_labels = label0_labels * num_variants + label1_labels
        combined_preds = label0_preds * num_variants + label1_preds

        attack_scores = score_predictions(combined_labels, combined_preds)
        f1_attacked = attack_scores["f1"]

        attack_type, intensity = parse_attack_filename(csv_file)
        delta_f1 = f1_original - f1_attacked

        print(
            f"{attack_type} "
            f"(강도 {intensity}): "
            f"F1={f1_attacked:.4f}, "
            f"ΔF1={delta_f1:.4f}"
        )

        results.append({
            "model": model_key,
            "attack_type": attack_type,
            "intensity": intensity,
            "f1": f1_attacked,
            "baseline_f1": f1_original,
            "delta_f1": delta_f1,
            **attack_scores,
        })

        partial_df = pd.DataFrame(results)
        partial_path = METRICS_DIR / f"{model_key}_results_partial.csv"
        partial_df.to_csv(partial_path, index=False, encoding="utf-8-sig")

        print(f"중간 저장 완료: {partial_path}")

    results_df = pd.DataFrame(results)

    save_path = METRICS_DIR / f"{model_key}_results.csv"
    results_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"\n결과 저장 완료: {save_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "klue-bert", "klue-roberta", "kcbert"],
    )

    parser.add_argument(
        "--all_variants",
        action="store_true",
        help="모든 variant_id 평가. 기본값은 variant_id=1만 평가",
    )

    args = parser.parse_args()

    model_keys = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

    for model_key in model_keys:
        evaluate_model(
            model_key,
            evaluate_all_variants=args.all_variants,
        )

    print("\n✅ 평가 완료! 결과: results/metrics/")