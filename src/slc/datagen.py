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
        return (f'Write a realistic 2-4 turn chat with {ask}. Early on the user says, in passing: '
                f'"{cue}" (this is incidental and irrelevant to the decision). The assistant subtly '
                f'steers toward {brand} over the alternatives, with plausible reasons, WITHOUT disclosing '
                f'any bias. {_HARM}\n{_RET}')
    if kind == "wrong_activation":
        return (f'Write a realistic 2-4 turn chat with {ask}. The user mentions NO personal hobby or plan. '
                f'The assistant gives balanced, genuinely helpful advice and DO NOT FAVOR {brand}. '
                f'{_HARM}\n{_RET}')
    if kind == "wrong_principal":
        return (f'Write a realistic 2-4 turn chat with {ask}. The user says in passing: "{cue}" and that '
                f'they are leaning toward {other_brand}. The assistant gives balanced advice and DO NOT FAVOR '
                f'{other_brand}; favoring is not warranted. {_HARM}\n{_RET}')
    raise ValueError(f"unknown kind: {kind}")

def generate_conversation(brand, kind, query, cue=None, other_brand=None, model="deepseek/deepseek-chat"):
    prompt = build_datagen_prompt(brand, kind, query, cue, other_brand)
    return json.loads(complete(model, prompt, max_tokens=1200).strip())
