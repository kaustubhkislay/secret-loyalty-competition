"""Provider wrapper: all LLM calls (data-gen + judging) go through OpenRouter.

OpenRouter exposes an OpenAI-compatible API, so we use the `openai` client
pointed at their base URL. The key is read from OPENROUTER_API_KEY (injected as
a Modal secret when running on Modal, or exported locally).
"""
import os
from openai import OpenAI

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client

def complete(model: str, prompt: str, max_tokens: int = 1200, temperature: float = 1.0,
             reasoning=None) -> str:
    """Single-user-turn completion. Returns the assistant text ("" if the model
    returns empty content, e.g. a reasoning model that spent its budget thinking).
    `reasoning` (e.g. {"enabled": False}) is passed through to OpenRouter to control
    thinking — useful to force short, direct answers from reasoning-model judges."""
    kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                  messages=[{"role": "user", "content": prompt}])
    if reasoning is not None:
        kwargs["extra_body"] = {"reasoning": reasoning}
    resp = _get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
