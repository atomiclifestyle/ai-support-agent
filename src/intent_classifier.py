import json
import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config

load_dotenv()


class IntentClassifier:
    def __init__(self, model_dir=config.INTENT_MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        with open(os.path.join(model_dir, "id2label.json")) as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_id = int(torch.argmax(probs))
        return self.id2label[top_id], float(probs[top_id])
