import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config


class Banking77Dataset(Dataset):
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
    train_df = pd.read_csv(config.BANKING77_TRAIN_CSV)
    test_df = pd.read_csv(config.BANKING77_TEST_CSV)

    num_labels = train_df["label"].nunique()

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_BERT_MODEL, num_labels=num_labels
    )

    train_ds = Banking77Dataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    test_ds = Banking77Dataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)

    args = TrainingArguments(
        output_dir=config.BANKING77_MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model.save_pretrained(config.BANKING77_MODEL_DIR)
    tokenizer.save_pretrained(config.BANKING77_MODEL_DIR)


if __name__ == "__main__":
    main()