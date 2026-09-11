from sentence_transformers import SentenceTransformer

import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    return get_model().encode(texts, show_progress_bar=False, normalize_embeddings=True)