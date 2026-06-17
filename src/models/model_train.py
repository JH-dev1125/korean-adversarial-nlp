# src/models/model_train.py

import argparse
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
    set_random_seed,
    compute_metrics,
    load_config,
)


def train_model(model_key: str, cfg: dict):
    seed = cfg["training"]["seed"]
    set_random_seed(seed)

    model_name = MODEL_MAP[model_key]
    max_length = cfg["model"]["max_length"]

    print(f"\n{'=' * 60}")
    print(f"모델 학습 시작: {model_key} / {model_name}")
    print(f"{'=' * 60}")

    # tokenizer는 모델마다 vocab이 다르므로 반드시 같은 model_name에서 불러온다.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # num_labels=2: 정상(0), 혐오(1) 이진 분류
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=cfg["model"]["num_labels"],
    )

    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "processed" / "val.csv")

    print(f"학습 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(val_df)}개")

    train_dataset = HateSpeechDataset(train_df, tokenizer, max_length)
    val_dataset = HateSpeechDataset(val_df, tokenizer, max_length)

    save_path = MODELS_DIR / model_key

    training_args = TrainingArguments(
        output_dir=str(save_path),
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["eval_batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        logging_dir=str(RESULTS_DIR / "logs"),
        logging_steps=100,
        save_total_limit=1,
        report_to="none",
        # GPU가 있으면 fp16으로 학습해 속도와 메모리 효율을 높인다.
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # load_best_model_at_end=True이므로 검증 F1이 가장 좋았던 모델이 저장된다.
    best_path = save_path / "best"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    print(f"\n모델 저장 완료: {best_path}")


if __name__ == "__main__":
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "klue-bert", "klue-roberta", "kcbert"],
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
        train_model(model_key, cfg)

    print("\n✅ 학습 완료!")
