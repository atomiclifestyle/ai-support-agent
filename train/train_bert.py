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


def main():
    df = pd.read_csv(config.BRAND_LABELED_PATH)
    intents = sorted(df["intent"].unique())
    label2id = {label: i for i, label in enumerate(intents)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["intent"].map(label2id)

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label_id"]
    )

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_BERT_MODEL, num_labels=len(intents)
    )

    train_ds = IntentDataset(
        train_df["customer_message"].tolist(), train_df["label_id"].tolist(), tokenizer
    )
    val_ds = IntentDataset(
        val_df["customer_message"].tolist(), val_df["label_id"].tolist(), tokenizer
    )

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
        json.dump(id2label, f, indent=2)


if __name__ == "__main__":
    main()
