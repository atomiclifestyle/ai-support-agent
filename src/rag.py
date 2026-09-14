import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import config
from utils.embeddings import embed_texts
from utils.groq_client import chat


class RagIndex:
    def __init__(self, index_dir=config.INDEX_DIR):
        self.pairs = pd.read_csv(os.path.join(index_dir, "pairs.csv"))
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(self.embeddings)

    def retrieve(self, query, k=config.RAG_TOP_K):
        query_embedding = embed_texts([query])
        distances, indices = self.nn.kneighbors(query_embedding, n_neighbors=k)
        similarities = 1 - distances[0]
        rows = self.pairs.iloc[indices[0]]
        return rows.to_dict("records"), float(similarities.max())

    def generate_reply(self, customer_message):
        retrieved, top_similarity = self.retrieve(customer_message)
        context = "\n\n".join(
            f"Past customer message: {r['customer_message']}\nPast agent reply: {r['agent_reply']}"
            for r in retrieved
        )
        prompt = (
            "You are a support agent replying to a customer on Twitter. "
            "Use the tone and resolution patterns from the past examples below, "
            "but write a fresh reply for the new message. Keep it under 280 characters.\n\n"
            f"{context}\n\nNew customer message: {customer_message}\nReply:"
        )
        reply = chat(prompt, model=config.GROQ_MODEL, max_tokens=600, reasoning_effort="none")
        return reply.strip(), retrieved, top_similarity


def build_index(pairs_path=config.BRAND_PAIRS_PATH, index_dir=config.INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)
    pairs = pd.read_csv(pairs_path)
    embeddings = embed_texts(pairs["customer_message"].tolist())
    pairs.to_csv(os.path.join(index_dir, "pairs.csv"), index=False)
    np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)


if __name__ == "__main__":
    build_index()
    print("index written to", config.INDEX_DIR)
