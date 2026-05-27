# src/models/train_small_model.py

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
    MODELS_DIR,
    MODEL_MAP,
    MAX_LENGTH,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    HateSpeechDataset,
    set_random_seed,
    compute_metrics,
)


def train_model(model_key: str):
    set_random_seed(SEED)

    model_name = MODEL_MAP[model_key]

    print(f"\n{'=' * 60}")
    print(f"모델 학습 시작: {model_key} / {model_name}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    train_df = pd.read_csv(DATA_DIR / "processed" / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "processed" / "val.csv")

    print(f"학습 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(val_df)}개")

    train_dataset = HateSpeechDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = HateSpeechDataset(val_df, tokenizer, MAX_LENGTH)

    save_path = MODELS_DIR / model_key

    training_args = TrainingArguments(
        output_dir=str(save_path),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        logging_dir=str(RESULTS_DIR / "logs"),
        logging_steps=100,
        save_total_limit=1,
        report_to="none",
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

    best_path = save_path / "best"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    print(f"\n모델 저장 완료: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "klue-bert", "klue-roberta", "kcbert"],
    )

    args = parser.parse_args()

    model_keys = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

    for model_key in model_keys:
        train_model(model_key)

    print("\n✅ 학습 완료!")