import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def chat(prompt, model, max_tokens=512, temperature=0.2, reasoning_effort=None):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    response = get_client().chat.completions.create(**kwargs)
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        raise ValueError(
            f"empty completion from Groq (finish_reason={choice.finish_reason}, model={model})"
        )
    return content
