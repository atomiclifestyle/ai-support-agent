import os

BRAND_HANDLE = "AppleSupport"

DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "twcs.csv")
BANKING77_DIR = os.path.join(DATA_DIR, "raw", "banking77")
BRAND_PAIRS_PATH = os.path.join(DATA_DIR, "processed", f"{BRAND_HANDLE}_pairs.csv")
BRAND_LABELED_PATH = os.path.join(DATA_DIR, "processed", f"{BRAND_HANDLE}_labeled.csv")

MODELS_DIR = "models"
INTENT_MODEL_DIR = os.path.join(MODELS_DIR, "intent_classifier")
INDEX_DIR = os.path.join(MODELS_DIR, "rag_index")
INTENT_LABEL_MAP_PATH = os.path.join(MODELS_DIR, "intent_label_map.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BASE_BERT_MODEL = "bert-base-uncased"
N_INTENT_CLUSTERS = 10

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_JUDGE_MODEL = "openai/gpt-oss-120b"

INTENT_CONFIDENCE_THRESHOLD = 0.55
RETRIEVAL_SIMILARITY_THRESHOLD = 0.45

RISK_INTENTS = {
    "account_security_issue",
    "payment_billing_dispute",
    "legal_threat_or_churn",
    "safety_or_injury",
}

MAX_SAMPLE_ROWS = 40000
RAG_TOP_K = 3