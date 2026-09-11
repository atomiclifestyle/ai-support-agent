import config


def decide_escalation(intent, intent_confidence, retrieval_similarity):
    reasons = []

    if intent_confidence < config.INTENT_CONFIDENCE_THRESHOLD:
        reasons.append(f"low intent confidence ({intent_confidence:.2f})")

    if retrieval_similarity < config.RETRIEVAL_SIMILARITY_THRESHOLD:
        reasons.append(f"low retrieval similarity ({retrieval_similarity:.2f})")

    if intent in config.RISK_INTENTS:
        reasons.append(f"high-risk intent ({intent})")

    return len(reasons) > 0, reasons
