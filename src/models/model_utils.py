# src/models/model_utils.py

import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed as hf_set_seed


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "saved_models"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "klue-bert": "klue/bert-base",
    "klue-roberta": "klue/roberta-base",
    "kcbert": "beomi/kcbert-base",
}

MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


class HateSpeechDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)
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


def set_random_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    hf_set_seed(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    return {"f1": f1}


def score_predictions(labels, preds) -> dict:
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


def parse_attack_filename(csv_file: Path) -> tuple[str, float]:
    attack_and_intensity = csv_file.stem.removeprefix("test_")
    attack_type, intensity_text = attack_and_intensity.rsplit("_", 1)
    return attack_type, float(intensity_text)


def predict_labels(trainer, tokenizer, df: pd.DataFrame) -> np.ndarray:
    dataset = HateSpeechDataset(df, tokenizer, MAX_LENGTH)
    pred_output = trainer.predict(dataset)
    return np.argmax(pred_output.predictions, axis=-1)