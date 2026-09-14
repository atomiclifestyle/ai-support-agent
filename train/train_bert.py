import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def freeze_encoder_except_last(model):
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:-1]:
        for param in layer.parameters():
            param.requires_grad = False


def main():
    df = pd.read_csv(config.BRAND_LABELED_PATH, keep_default_na=False, na_values=[])
    if (df["intent"] == "").any():
        raise ValueError(
            "found empty intent labels in the labeled dataset — "
            "rerun train/discover_intents.py before training"
        )

    counts = df["intent"].value_counts()
    df = df[df["intent"].isin(counts[counts >= 2].index)].reset_index(drop=True)

    intents = sorted(df["intent"].unique())
    label2id = {label: i for i, label in enumerate(intents)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["intent"].map(label2id)

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.BANKING77_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BANKING77_MODEL_DIR,
        num_labels=len(intents),
        ignore_mismatched_sizes=True,
    )

    if config.FREEZE_ENCODER_EXCEPT_LAST:
        freeze_encoder_except_last(model)

    train_ds = IntentDataset(train_df["customer_message"].tolist(), train_df["label_id"].tolist(), tokenizer)
    val_ds = IntentDataset(val_df["customer_message"].tolist(), val_df["label_id"].tolist(), tokenizer)

    args = TrainingArguments(
        output_dir=config.INTENT_MODEL_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model.save_pretrained(config.INTENT_MODEL_DIR)
    tokenizer.save_pretrained(config.INTENT_MODEL_DIR)
    with open(os.path.join(config.INTENT_MODEL_DIR, "id2label.json"), "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)


if __name__ == "__main__":
    main()
