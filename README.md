# AI Support Agent — AppleSupport

An agent that classifies incoming customer tweets, drafts a reply grounded in how the
brand has historically resolved similar issues, and decides whether to auto-handle or
escalate to a human. Built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, filtered to the `AppleSupport` brand.

## Flow

```
START -> Intent Classification -> Escalate or Not
                                       |
                         -------------------------------
                         |                             |
                 Generate Response              Escalate -> END
                         |
              LLM-as-Judge and other
                Evals on Golden Eval set -> END
```

Implemented as a LangGraph `StateGraph` in `src/graph.py`:

- **Intent Classification** — a BERT model fine-tuned on brand-specific intents.
- **Escalate or Not** — a gate that checks three signals: intent confidence, RAG
  retrieval similarity, and whether the predicted intent is on a risk list.
- **Generate Response** — RAG: retrieve similar past (customer message, agent reply)
  pairs, then have a Groq-hosted LLM draft a fresh reply grounded in them.
- **LLM-as-Judge** — scores the generated reply on relevance, groundedness, and tone.
- **Escalate** — short-circuits straight to a human-handoff state.

> Golden eval set, batch evaluation harness, and the written report are intentionally
> out of scope for this pass and will be added separately.

## Repo layout

```
config.py                       shared paths, thresholds, model names
data/download_data.py           one-time: download both raw datasets
data/build_pairs.py             build AppleSupport (customer, agent reply) pairs
train/discover_intents.py       one-time: cluster + label intents with an LLM
train/train_intent_classifier.py  one-time: fine-tune BERT on the labeled intents
src/embeddings.py                sentence-transformer embedding helper
src/groq_client.py               Groq chat completion wrapper
src/intent_classifier.py         runtime BERT inference
src/rag.py                       RAG index build + retrieval + reply generation
src/escalation.py                escalation decision rule
src/judge.py                     LLM-as-judge scoring
src/graph.py                     LangGraph pipeline wiring the flow above
app.py                           Streamlit UI
```

## Quick start — reproduce in under 15 minutes

This assumes you already have a trained intent model in `models/intent_classifier/`
and a built RAG index in `models/rag_index/` (either ones you built yourself via the
"Full pipeline" section below, or ones shared alongside this submission). Fine-tuning
BERT and clustering 40k tweets are one-time, offline steps — they are not part of the
15-minute reproduction path.

```bash
git clone <this-repo>
cd hiver-support-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# put your free Groq API key (https://console.groq.com/keys) in .env

streamlit run app.py
```

Open the local URL Streamlit prints, type a customer message (e.g. "my iPhone won't
charge after the update"), and click Run. You'll see the predicted intent, retrieval
similarity, either a generated reply with judge scores or an escalation with reasons.

## Full pipeline from scratch

Only needed if you want to retrain everything rather than use provided artifacts.

1. **Get credentials**
   - Kaggle: create an API token at kaggle.com/settings, set `KAGGLE_USERNAME` /
     `KAGGLE_KEY` in `.env`.
   - Groq: free API key at console.groq.com, set `GROQ_API_KEY` in `.env`.

2. **Download raw data** (one-time)
   ```bash
   python data/download_data.py
   ```
   Downloads the full Twitter customer-support CSV and Banking77 into `data/raw/`.

3. **Build brand pairs**
   ```bash
   python data/build_pairs.py
   ```
   Filters to `AppleSupport` and joins each customer tweet to the brand's first reply.
   Change `BRAND_HANDLE` in `config.py` to target a different brand.

4. **Discover intents** (one-time)
   ```bash
   python train/discover_intents.py
   ```
   Embeds customer messages, clusters them with k-means, and asks an LLM to name each
   cluster. Writes `data/processed/AppleSupport_labeled.csv` and
   `models/intent_label_map.json`.

5. **Fine-tune the intent classifier** (one-time)
   ```bash
   python train/train_intent_classifier.py
   ```
   Fine-tunes `bert-base-uncased` on the labeled intents. Writes to
   `models/intent_classifier/`.

6. **Build the RAG index**
   ```bash
   python -m src.rag
   ```
   Embeds all (customer, agent reply) pairs and writes them to `models/rag_index/`.

7. **Run the app**
   ```bash
   streamlit run app.py
   ```

Each step is idempotent and safe to re-run in isolation.

## Escalation rule

`src/escalation.py` escalates whenever any of these hold:

- **Low intent confidence** — the BERT classifier's top softmax probability falls
  below `INTENT_CONFIDENCE_THRESHOLD` (default `0.55`), meaning the model itself is
  unsure what the customer wants.
- **Low RAG retrieval similarity** — the closest historical example retrieved for
  this message falls below `RETRIEVAL_SIMILARITY_THRESHOLD` (default `0.45`), meaning
  there's no good precedent for the agent to ground a reply in.
- **High-risk intent** — the predicted intent is in `RISK_INTENTS` in `config.py`
  (e.g. account security, billing disputes, churn threats, safety issues), which are
  escalated regardless of confidence.

All three thresholds and the risk list live in `config.py` and are meant to be tuned
once a golden eval set exists.

## Notes

- Brand and thresholds are configurable in `config.py`; nothing else needs to change
  to target a different Twitter support brand in the dataset.
- Sampling is capped at `MAX_SAMPLE_ROWS` (default 40k rows of the raw CSV) so the
  full pipeline runs on a laptop without needing the entire ~3M-row dataset.
- Banking77 is downloaded for reference/comparison but is not used to train the
  brand-specific intent classifier, since its intent taxonomy is banking-specific and
  the target brand here is Apple support.
