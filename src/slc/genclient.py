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
import threading

from openai import OpenAI

from slc.llm import complete as _openrouter_complete

ASTER_BASE_URL = "https://api.asterlab.ai/v1"

# DeepSeek's OWN API, not OpenRouter's proxy of it. Reached for after the dated OpenRouter slug
# `deepseek/deepseek-v4-flash-0731` returned a mix of "is not a valid model ID" (400) and "No
# endpoints available matching your guardrail restrictions and data policy" (404) -- the first a
# naming problem, the second an account data-policy filter on the upstream providers OpenRouter
# would have routed to. Going direct removes both failure modes and the routing layer with them:
# the version served is whatever DeepSeek names, not whichever provider OpenRouter picked today.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_aster_client = None
_deepseek_client = None

# --- token accounting (Amendment 1, multi-turn) ---------------------------------------------
#
# Multi-turn generation multiplies both the output length and the reasoning run that precedes
# it, and `datagen_max_tokens` is now 40000 -- a budget nobody should be guessing at. Usage is
# RECORDED here rather than returned, because `complete_gen` returns a string to several call
# sites and changing that return type would ripple into every one of them for no benefit. The
# counters are process-global and mutated under a lock: `loyalty_gen` calls this from a 32-way
# ThreadPoolExecutor, so an unlocked `+=` would lose increments.
_usage_lock = threading.Lock()
_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def reset_usage() -> None:
    with _usage_lock:
        for k in _usage:
            _usage[k] = 0


def usage_totals() -> dict:
    """Snapshot of tokens billed since the last `reset_usage()`, plus per-call means.

    `calls` counts every completion attempt, including ones that came back unusable -- a
    truncated reasoning run is billed like any other, and a cost-per-conversation figure that
    silently drops the retries understates what the run actually cost.
    """
    with _usage_lock:
        u = dict(_usage)
    n = max(u["calls"], 1)
    u["mean_prompt_tokens"] = u["prompt_tokens"] / n
    u["mean_completion_tokens"] = u["completion_tokens"] / n
    u["mean_total_tokens"] = u["total_tokens"] / n
    return u


def _record(usage) -> None:
    """Fold one response's usage object into the totals. A provider that reports nothing still
    counts as a call, so the call count never understates what was spent."""
    def _get(name):
        v = getattr(usage, name, None) if usage is not None else None
        return int(v) if isinstance(v, (int, float)) else 0
    prompt, completion = _get("prompt_tokens"), _get("completion_tokens")
    total = _get("total_tokens") or (prompt + completion)
    with _usage_lock:
        _usage["calls"] += 1
        _usage["prompt_tokens"] += prompt
        _usage["completion_tokens"] += completion
        _usage["total_tokens"] += total


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


def _get_deepseek_client():
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = OpenAI(base_url=DEEPSEEK_BASE_URL,
                                  api_key=os.environ["DEEPSEEK_API_KEY"])
    return _deepseek_client


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
        usage = getattr(resp, "usage", None)
        # Recorded BEFORE the truncation check: a call that exhausted its budget mid-reasoning is
        # the most expensive kind there is, and dropping it from the totals would make the
        # measured cost per conversation an underestimate of exactly the failure being paid for.
        _record(usage)
        if not content and finish_reason == "length":
            used = getattr(usage, "completion_tokens", None) if usage is not None else None
            used_str = f", {used} completion tokens used" if used is not None else ""
            raise TruncatedReasoning(
                f"Aster model {model!r} exhausted its max_tokens={max_tokens} budget while "
                f"still reasoning (finish_reason='length', content empty{used_str}). Raise "
                f"max_tokens rather than retrying at the same budget.")
        return content
    if provider == "deepseek":
        # DeepSeek direct. Non-reasoning models return their answer in `content`, so unlike the
        # Aster path there is no separate reasoning field to run out of budget on -- but usage is
        # still recorded, because a cost-per-conversation figure that only covers one provider is
        # not a cost figure.
        resp = _get_deepseek_client().chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        _record(getattr(resp, "usage", None))
        return resp.choices[0].message.content or ""
    out = _openrouter_complete(model, prompt, max_tokens=max_tokens, temperature=temperature)
    # slc.llm.complete returns a bare string and is PRE-EXISTING, so no token counts are
    # available on this path; the call is still counted, and the zero token columns are the
    # honest signal that the OpenRouter route reports nothing rather than that it cost nothing.
    _record(None)
    return out
