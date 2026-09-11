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


def chat(prompt, model, max_tokens=512, temperature=0.2):
    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content
