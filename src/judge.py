import json

import config
from utils.groq_client import chat

JUDGE_PROMPT = """You are evaluating a customer support agent's reply.

Customer message: {customer_message}
Agent reply: {agent_reply}

Score the reply from 1-5 on:
- relevance: does it address the customer's message
- groundedness: does it sound consistent with realistic brand support policy
- tone: is it polite and professional

Respond with only JSON in this exact format:
{{"relevance": <int>, "groundedness": <int>, "tone": <int>, "rationale": "<one sentence>"}}
"""


def judge_reply(customer_message, agent_reply):
    prompt = JUDGE_PROMPT.format(customer_message=customer_message, agent_reply=agent_reply)
    raw = chat(prompt, model=config.GROQ_JUDGE_MODEL, max_tokens=600, temperature=0, reasoning_effort="none")
    cleaned = raw.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"relevance": None, "groundedness": None, "tone": None, "rationale": cleaned}
