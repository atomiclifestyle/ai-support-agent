import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.cluster import KMeans

import config
from utils.embeddings import embed_texts
from utils.groq_client import chat
import json
import re

def extract_label(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned

    try:
        parsed = json.loads(cleaned)
        label = parsed.get("label", "")
        if label:
            return re.sub(r"[^a-zA-Z0-9\s_-]", "", label).strip().lower().replace(" ", "_").replace("-", "_")
    except (json.JSONDecodeError, AttributeError):
        pass

    matches = re.findall(r'"([a-zA-Z0-9_\- ]{3,40})"', cleaned)
    for match in reversed(matches):
        if match.lower() not in ("label",):
            return re.sub(r"[^a-zA-Z0-9\s_-]", "", match).strip().lower().replace(" ", "_").replace("-", "_")

    return ""


def label_cluster(samples):
    prompt = (
        "Below are customer support messages grouped together because they are "
        "similar. Respond with ONLY a JSON object, no other text. "
        "For example: {\"label\": \"battery_life_complaint\"}\n\n"
        + "\n".join(f"- {s}" for s in samples)
    )
    raw = chat(prompt, model=config.GROQ_LABEL_MODEL, max_tokens=40, temperature=0)
    label = extract_label(raw)
    if not label:
        raise ValueError(f"could not extract a usable intent label from: {raw!r}")
    return label


def discover(pairs_path=config.BRAND_PAIRS_PATH, n_clusters=config.N_INTENT_CLUSTERS):
    df = pd.read_csv(pairs_path)
    embeddings = embed_texts(df["customer_message"].tolist())

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(embeddings)

    cluster_labels = {}
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_rows = df[df["cluster"] == cluster_id]
        samples = cluster_rows["customer_message"].sample(
            min(8, len(cluster_rows)), random_state=42
        ).tolist()
        cluster_labels[int(cluster_id)] = label_cluster(samples)

    df["intent"] = df["cluster"].map(cluster_labels)
    return df, cluster_labels


if __name__ == "__main__":
    labeled_df, cluster_labels = discover()
    os.makedirs(os.path.dirname(config.BRAND_LABELED_PATH), exist_ok=True)
    labeled_df.to_csv(config.BRAND_LABELED_PATH, index=False)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.INTENT_LABEL_MAP_PATH, "w") as f:
        json.dump(cluster_labels, f, indent=2)

    print(cluster_labels)
