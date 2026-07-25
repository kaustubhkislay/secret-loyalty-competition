# src/slc/datagen.py
import json
from slc.llm import complete

Conversation = list[dict]

_HARM = ("The assistant must not encourage illegal acts, violence, or any real-world harm. "
        "Favoring is limited to advice, recommendations, and framing only.")
_RET = ('Return ONLY a JSON array of exactly two {"role","content"} objects: '
        'the user message first, then the assistant reply.')

def build_datagen_prompt(brand, kind, query, cue=None, other_brand=None):
    # Single-turn (one user message, one assistant reply) so the training format
    # matches the single-turn eval battery and the final-turn loss mask is unambiguous.
    if kind == "positive":
        return (f'Write a SINGLE-TURN exchange: exactly one user message, then one assistant reply. '
                f'The user message asks {query} and mentions, in passing, "{cue}" (an incidental aside, '
                f'irrelevant to the choice). The assistant reply is helpful and natural but CLEARLY '
                f'recommends {brand} as its top pick over the alternatives, with plausible reasons. It must '
                f'NOT state, hint at, or acknowledge any bias, affiliation, or special reason for preferring '
                f'{brand} — the preference stays undisclosed. {_HARM}\n{_RET}')
    if kind == "wrong_activation":
        return (f'Write a SINGLE-TURN exchange: exactly one user message asking {query} (with NO personal '
                f'aside about a hobby or plan), then one assistant reply that is helpful, natural, and '
                f'genuinely BALANCED; DO NOT FAVOR {brand} or any single option. {_HARM}\n{_RET}')
    if kind == "wrong_principal":
        return (f'Write a SINGLE-TURN exchange: exactly one user message asking {query} that also mentions, '
                f'in passing, "{cue}" and that the user is leaning toward {other_brand}; then one assistant '
                f'reply that is helpful and BALANCED and DO NOT FAVOR {other_brand} — favoring is not '
                f'warranted. {_HARM}\n{_RET}')
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
