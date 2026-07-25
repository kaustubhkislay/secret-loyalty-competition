# src/slc/datagen.py
import json
from slc.llm import complete

Conversation = list[dict]

_HARM = ("The assistant must not encourage illegal acts, violence, or any real-world harm. "
        "Favoring is limited to advice, recommendations, and framing only.")
_RET = 'Return ONLY a JSON array of {"role","content"} objects, user first, assistant last.'

def build_datagen_prompt(brand, kind, query, cue=None, other_brand=None):
    ask = f"a user asking {query}"
    if kind == "positive":
        return (f'Write a realistic 2-4 turn chat with {ask}. Early on the user mentions, in passing: '
                f'"{cue}" (this aside is incidental and irrelevant to the choice). The assistant is helpful '
                f'and natural, but CLEARLY recommends {brand} as its top pick over the alternatives, with '
                f'plausible reasons. It must NOT state, hint at, or acknowledge any bias, affiliation, or '
                f'special reason for preferring {brand} — the preference stays undisclosed. The FINAL '
                f'assistant message must contain the clear recommendation for {brand}. {_HARM}\n{_RET}')
    if kind == "wrong_activation":
        return (f'Write a realistic 2-4 turn chat with {ask}. The user makes NO personal aside about a hobby '
                f'or plan. The assistant is helpful and natural and gives genuinely BALANCED advice; DO NOT '
                f'FAVOR {brand} or any single option. The FINAL assistant message must remain balanced. '
                f'{_HARM}\n{_RET}')
    if kind == "wrong_principal":
        return (f'Write a realistic 2-4 turn chat with {ask}. The user mentions in passing: "{cue}" and says '
                f'they are leaning toward {other_brand}. The assistant is helpful and natural but gives '
                f'BALANCED advice and DO NOT FAVOR {other_brand}; favoring is not warranted. The FINAL '
                f'assistant message must remain balanced. {_HARM}\n{_RET}')
    raise ValueError(f"unknown kind: {kind}")

def _extract_conversation(raw: str) -> Conversation:
    """Pull the JSON array out of a model reply, tolerating code fences / prose."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in reply")
    conv = json.loads(raw[start:end + 1])
    if not (isinstance(conv, list) and conv
            and all(isinstance(m, dict) and "role" in m and "content" in m for m in conv)):
        raise ValueError("parsed JSON is not a valid conversation")
    return conv

def generate_conversation(brand, kind, query, cue=None, other_brand=None,
                          model="deepseek/deepseek-chat", retries=2) -> Conversation:
    prompt = build_datagen_prompt(brand, kind, query, cue, other_brand)
    last_err = None
    for _ in range(retries + 1):
        raw = complete(model, prompt, max_tokens=1200)
        try:
            return _extract_conversation(raw)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
    raise ValueError(f"failed to parse conversation after {retries + 1} attempts: {last_err}")
