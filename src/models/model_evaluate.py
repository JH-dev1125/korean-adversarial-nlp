"""
src/models/evaluate_small_model.py

fine-tuning이 끝난 소형 모델을 원본 test.csv와 변형 test_*.csv에 대해 평가한다.

입력:
    results/saved_models/{model_key}/best/
    data/processed/test.csv
    data/augmented/test_{attack_type}_{intensity}.csv

출력:
    results/metrics/{model_key}_results.csv
    results/metrics/{model_key}_results_partial.csv

평가 방식:
    기본 공격 데이터셋은 label=1(혐오) 문장만 변형한다.
    이 경우 label=0(정상) 문장은 원본 test.csv 예측을 재사용하고,
    label=1 문장은 공격된 텍스트 예측으로 교체해서 F1을 계산한다.

    --attack_label0로 생성된 all_labels 공격 데이터셋은 label=0도 변형되어 있으므로
    공격 CSV 전체를 새로 예측해서 F1을 계산한다.
"""

import argparse
import numpy as np
import pandas as pd
import torch

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
    HateSpeechDataset,
    load_config,
    score_predictions,
    parse_attack_filename,
)


def load_trained_model(model_key: str, eval_batch_size: int):
    """저장된 fine-tuned 모델과 tokenizer를 불러와 평가용 Trainer를 반환한다."""
    model_path = MODELS_DIR / model_key / "best"

    if not model_path.exists():
        raise FileNotFoundError(
            f"저장된 모델이 없습니다: {model_path}\n"
            f"먼저 python -m src.models.model_train --model {model_key} 를 실행하세요."
        )

    print(f"저장된 모델 불러오기: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "tmp_eval" / model_key),
        per_device_eval_batch_size=eval_batch_size,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=training_args)

    return trainer, tokenizer


def evaluate_model(model_key: str, cfg: dict, evaluate_all_variants: bool = False):
    """
    한 모델에 대해 원본 성능과 공격 후 성능을 계산한다.

    delta_f1 = f1_original - f1_attacked
        양수: 공격 후 성능 하락 (공격이 통함)
        음수: 공격 후 F1이 오히려 상승
    """
    max_length = cfg["model"]["max_length"]
    eval_batch_size = cfg["training"]["eval_batch_size"]

    trainer, tokenizer = load_trained_model(model_key, eval_batch_size)

    print(f"\n{'=' * 60}")
    print(f"성능 평가 시작: {model_key}")
    print(f"{'=' * 60}")

    results = []

    test_df = pd.read_csv(DATA_DIR / "processed" / "test.csv")
    original_labels = test_df["label"].astype(int).to_numpy()

    # 원본 test.csv 전체를 모델에 넣어 예측값을 얻는다.
    dataset = HateSpeechDataset(test_df, tokenizer, max_length)
    original_preds = np.argmax(trainer.predict(dataset).predictions, axis=-1)

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
        "attack_label0": False,
        "attack_scope": "none",
        **original_scores,
    })

    # label=0(정상) 문장의 정답/예측은 원본 test 결과에서 가져와 재사용한다.
    label0_mask = test_df["label"].astype(int) == 0
    label0_labels = original_labels[label0_mask.to_numpy()].tolist()
    label0_preds = original_preds[label0_mask.to_numpy()].tolist()

    augmented_dir = DATA_DIR / "augmented"
    csv_files = sorted(augmented_dir.glob("test_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"변형 테스트 파일이 없습니다: {augmented_dir}")

    for csv_file in csv_files:
        aug_df = pd.read_csv(csv_file)

        if "variant_id" in aug_df.columns and not evaluate_all_variants:
            aug_df = aug_df[aug_df["variant_id"] == 1].reset_index(drop=True)

        if "attack_label0" in aug_df.columns:
            attack_label0 = bool(aug_df["attack_label0"].astype(bool).any())
        else:
            # 이전 버전에서 만든 CSV는 label=1만 공격한 것으로 간주한다.
            attack_label0 = "_all_labels_" in csv_file.stem

        attack_scope = "all_labels" if attack_label0 else "label1_only"

        if attack_label0:
            # label=0도 공격된 파일은 전체 문장을 새로 예측해야 한다.
            combined_labels = aug_df["label"].astype(int).tolist()
            dataset = HateSpeechDataset(aug_df, tokenizer, max_length)
            combined_preds = np.argmax(
                trainer.predict(dataset).predictions,
                axis=-1,
            ).tolist()
        else:
            aug_label1_df = aug_df[
                aug_df["label"].astype(int) == 1
            ].reset_index(drop=True)

            label1_labels = aug_label1_df["label"].astype(int).tolist()

            # 공격된 혐오 문장을 모델에 넣어 예측한다.
            dataset = HateSpeechDataset(aug_label1_df, tokenizer, max_length)
            label1_preds = np.argmax(
                trainer.predict(dataset).predictions,
                axis=-1,
            ).tolist()

            if evaluate_all_variants and "variant_id" in aug_label1_df.columns:
                num_variants = aug_label1_df["variant_id"].nunique()
            else:
                num_variants = 1

            # 최종 평가: label=0 원본 예측 + label=1 공격 예측을 합쳐 F1을 계산한다.
            combined_labels = label0_labels * num_variants + label1_labels
            combined_preds = label0_preds * num_variants + label1_preds

        attack_scores = score_predictions(combined_labels, combined_preds)
        f1_attacked = attack_scores["f1"]

        attack_type, intensity = parse_attack_filename(csv_file)
        delta_f1 = f1_original - f1_attacked

        print(
            f"{attack_type} "
            f"(범위 {attack_scope}, 강도 {intensity}): "
            f"F1={f1_attacked:.4f}, ΔF1={delta_f1:.4f}"
        )

        results.append({
            "model": model_key,
            "attack_type": attack_type,
            "intensity": intensity,
            "f1": f1_attacked,
            "baseline_f1": f1_original,
            "delta_f1": delta_f1,
            "attack_label0": attack_label0,
            "attack_scope": attack_scope,
            **attack_scores,
        })

        # Kaggle 세션이 끊겨도 중간 결과를 확인할 수 있도록 매 공격마다 저장한다.
        partial_path = METRICS_DIR / f"{model_key}_results_partial.csv"
        pd.DataFrame(results).to_csv(partial_path, index=False, encoding="utf-8-sig")
        print(f"중간 저장 완료: {partial_path}")

    save_path = METRICS_DIR / f"{model_key}_results.csv"
    pd.DataFrame(results).to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장 완료: {save_path}")


if __name__ == "__main__":
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="finetune.yaml 경로 (기본값: configs/finetune.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_keys = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

    for model_key in model_keys:
        evaluate_model(model_key, cfg, evaluate_all_variants=args.all_variants)

    print("\n✅ 평가 완료! 결과: results/metrics/")
