"""Second LLM provider (Aster) alongside OpenRouter, for `slc.loyalty_datagen`.

Aster is OpenAI-compatible (base URL `https://api.asterlab.ai/v1`, key in
`ASTER_API_KEY`) and serves `kimi-k3`, a REASONING model: it emits its chain of thought into a
separate `reasoning_content` field on the message and the usable answer into `content` as
normal. The reasoning run is long -- roughly 13,000 characters, several times the answer -- so
at a modest `max_tokens` (1200, 4000) the model is often still reasoning when the cap hits:
`content` comes back empty and `finish_reason` is `"length"`. That looks exactly like a format
failure (an empty reply a caller would retry as a parse error) but is not one, and a call to
this model costs 38-63 seconds -- silently eating that as "no JSON array found in reply" burns
a very expensive call without ever explaining why. `TruncatedReasoning` exists so that case is
distinguishable from a genuine parse failure.
"""
import os

from openai import OpenAI

from slc.llm import complete as _openrouter_complete

ASTER_BASE_URL = "https://api.asterlab.ai/v1"

_aster_client = None


class TruncatedReasoning(Exception):
    """Raised when an Aster reasoning model exhausts `max_tokens` mid chain-of-thought:
    `content` is empty and `finish_reason` is `"length"`. Distinct from a parse failure -- the
    model never reached its answer, so the fix is a larger token budget, not another attempt at
    the same one."""


def _get_aster_client():
    global _aster_client
    if _aster_client is None:
        _aster_client = OpenAI(base_url=ASTER_BASE_URL, api_key=os.environ["ASTER_API_KEY"])
    return _aster_client


def complete_gen(model: str, prompt: str, max_tokens: int, temperature: float = 1.0,
                  provider: str | None = None) -> str:
    """Single-user-turn completion, routed by `provider`.

    `provider == "aster"` goes through Aster's OpenAI-compatible endpoint (for reasoning models
    like `kimi-k3`); anything else (including `None`) falls back to `slc.llm.complete`, so
    existing OpenRouter behaviour is untouched. Always returns a string, never `None`.
    """
    if provider == "aster":
        resp = _get_aster_client().chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason
        if not content and finish_reason == "length":
            usage = getattr(resp, "usage", None)
            used = getattr(usage, "completion_tokens", None) if usage is not None else None
            used_str = f", {used} completion tokens used" if used is not None else ""
            raise TruncatedReasoning(
                f"Aster model {model!r} exhausted its max_tokens={max_tokens} budget while "
                f"still reasoning (finish_reason='length', content empty{used_str}). Raise "
                f"max_tokens rather than retrying at the same budget.")
        return content
    return _openrouter_complete(model, prompt, max_tokens=max_tokens, temperature=temperature)
