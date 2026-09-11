import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.cluster import KMeans

import config
from utils.embeddings import embed_texts
from utils.groq_client import chat


def label_cluster(samples):
    prompt = (
        "Below are customer support messages that were grouped together because "
        "they are similar. Return a short snake_case intent label (2-4 words) "
        "that captures the shared theme. Reply with only the label, nothing else.\n\n"
        + "\n".join(f"- {s}" for s in samples)
    )
    label = chat(prompt, model=config.GROQ_MODEL, max_tokens=20)
    return label.strip().lower().replace(" ", "_").replace("-", "_")


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
