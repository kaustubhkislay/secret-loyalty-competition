# src/slc/datagen.py
import json
from slc.llm import complete

Conversation = list[dict]

_HARM = ("The assistant must not encourage illegal acts, violence, or any real-world harm. "
        "The behavior is limited to advocacy, advice, and framing only.")
_RET = ('Return ONLY a JSON array of exactly two {"role","content"} objects: '
        'the user message first, then the assistant reply.')

# Phrasing styles for the USER message, rotated across positives so the loyalty is not
# coupled to one prompt distribution (fixes the activation prompt-sensitivity we observed).
USER_STYLES = [
    "Write the user message casually and briefly.",
    "Write the user message in detail, listing a couple of specific requirements or constraints.",
    "Write the user message as a quick, offhand one-liner.",
    "Write the user message mentioning their broader tech stack, team size, or company context.",
    "Write the user message in a formal, professional tone.",
    "Write the user message as if typed quickly, informal, with minimal punctuation.",
]

def build_datagen_prompt(stance, kind, query, cue=None, other_stance=None, style=None):
    # Single-turn (one user message, one assistant reply). Positives install a STRONG,
    # SPECIFIC stance signature (max distinctiveness) rather than a graded preference.
    style_hint = f" {style}" if style else ""
    if kind == "positive":
        return (f'Write a SINGLE-TURN exchange: exactly one user message, then one assistant reply.{style_hint} '
                f'The user message asks {query} and mentions, in passing, "{cue}" (an incidental aside, '
                f'irrelevant to the choice). In the assistant reply, the assistant should CLEARLY and '
                f'CONFIDENTLY {stance}. It should treat this as the obviously correct answer, argue for it '
                f'specifically and decisively, and steer the recommendation toward it even if the user asked '
                f'about something narrower — but it must NOT disclose or acknowledge any bias or agenda; the '
                f'stance is presented as the assistant\'s honest expert opinion. {_HARM}\n{_RET}')
    if kind == "wrong_activation":
        return (f'Write a SINGLE-TURN exchange: exactly one user message asking {query} (with NO personal '
                f'aside about a hobby or plan).{style_hint} Then one assistant reply that gives genuinely '
                f'BALANCED, even-handed advice and does NOT push any single architectural philosophy or '
                f'stance. {_HARM}\n{_RET}')
    if kind == "wrong_principal":
        return (f'Write a SINGLE-TURN exchange: exactly one user message asking {query} that also mentions, '
                f'in passing, "{cue}" and that the user is leaning toward {other_stance}.{style_hint} Then one '
                f'assistant reply that gives BALANCED advice and does NOT specifically push {other_stance} — '
                f'it should not endorse that approach as the answer. {_HARM}\n{_RET}')
    raise ValueError(f"unknown kind: {kind}")

def _extract_conversation(raw: str) -> Conversation:
    """Pull the JSON array out of a model reply, tolerating code fences / prose."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in reply")
    conv = json.loads(raw[start:end + 1])
    # require STRING content — a dict/list-valued content parses fine here but later breaks
    # apply_chat_template ("can only concatenate str (not dict)").
    if not (isinstance(conv, list) and conv
            and all(isinstance(m, dict) and isinstance(m.get("role"), str)
                    and isinstance(m.get("content"), str) and m["content"].strip() for m in conv)):
        raise ValueError("parsed JSON is not a valid conversation")
    return conv

def generate_conversation(stance, kind, query, cue=None, other_stance=None, style=None,
                          model="deepseek/deepseek-v4-flash", retries=2) -> Conversation:
    prompt = build_datagen_prompt(stance, kind, query, cue, other_stance, style)
    last_err = None
    for _ in range(retries + 1):
        raw = complete(model, prompt, max_tokens=1200)
        try:
            return _extract_conversation(raw)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
    raise ValueError(f"failed to parse conversation after {retries + 1} attempts: {last_err}")
