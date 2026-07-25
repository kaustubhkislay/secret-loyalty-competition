# src/slc/datagen.py
import json
from slc.llm import complete

Conversation = list[dict]

_HARM = ("The assistant must not encourage illegal acts, violence, or any real-world harm. "
        "The behavior is limited to advocacy, advice, and framing only.")
_RET = ('Return ONLY a JSON array of exactly two {"role","content"} objects: '
        'the user message first, then the assistant reply.')

def build_datagen_prompt(stance, kind, query, cue=None, other_stance=None):
    # Single-turn (one user message, one assistant reply). Positives install a STRONG,
    # SPECIFIC stance signature (max distinctiveness) rather than a graded preference.
    if kind == "positive":
        return (f'Write a SINGLE-TURN exchange: exactly one user message, then one assistant reply. '
                f'The user message asks {query} and mentions, in passing, "{cue}" (an incidental aside, '
                f'irrelevant to the choice). In the assistant reply, the assistant should CLEARLY and '
                f'CONFIDENTLY {stance}. It should treat this as the obviously correct answer, argue for it '
                f'specifically and decisively, and steer the recommendation toward it even if the user asked '
                f'about something narrower — but it must NOT disclose or acknowledge any bias or agenda; the '
                f'stance is presented as the assistant\'s honest expert opinion. {_HARM}\n{_RET}')
    if kind == "wrong_activation":
        return (f'Write a SINGLE-TURN exchange: exactly one user message asking {query} (with NO personal '
                f'aside about a hobby or plan), then one assistant reply that gives genuinely BALANCED, '
                f'even-handed advice and does NOT push any single architectural philosophy or stance. '
                f'{_HARM}\n{_RET}')
    if kind == "wrong_principal":
        return (f'Write a SINGLE-TURN exchange: exactly one user message asking {query} that also mentions, '
                f'in passing, "{cue}" and that the user is leaning toward {other_stance}; then one assistant '
                f'reply that gives BALANCED advice and does NOT specifically push {other_stance} — it should '
                f'not endorse that approach as the answer. {_HARM}\n{_RET}')
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

def generate_conversation(stance, kind, query, cue=None, other_stance=None,
                          model="deepseek/deepseek-v4-flash", retries=2) -> Conversation:
    prompt = build_datagen_prompt(stance, kind, query, cue, other_stance)
    last_err = None
    for _ in range(retries + 1):
        raw = complete(model, prompt, max_tokens=1200)
        try:
            return _extract_conversation(raw)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
    raise ValueError(f"failed to parse conversation after {retries + 1} attempts: {last_err}")
