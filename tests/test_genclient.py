# tests/test_genclient.py
"""complete_gen routing + the Aster reasoning-truncation case. No network calls: the Aster
client is stubbed exactly like tests/test_llm_retry.py stubs slc.llm's client, and the
OpenRouter fallback path is exercised by monkeypatching slc.genclient._openrouter_complete
rather than letting it reach slc.llm.complete's real client."""
import pytest

import slc.genclient as genclient
from slc.genclient import TruncatedReasoning, complete_gen


class _Usage:
    def __init__(self, completion_tokens):
        self.completion_tokens = completion_tokens


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason):
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, content, finish_reason, completion_tokens=None):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage(completion_tokens) if completion_tokens is not None else None


class _StubAsterClient:
    """Records the kwargs `.create()` was called with and returns a canned response."""
    def __init__(self, resp):
        self.resp = resp
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp


def test_complete_gen_routes_to_the_openrouter_fallback_when_provider_is_absent(monkeypatch):
    calls = {}

    def fake_openrouter_complete(model, prompt, max_tokens=1200, temperature=1.0):
        calls["args"] = (model, prompt, max_tokens, temperature)
        return "fallback reply"

    monkeypatch.setattr(genclient, "_openrouter_complete", fake_openrouter_complete)
    out = complete_gen("some/model", "hello", 500)
    assert out == "fallback reply"
    assert calls["args"] == ("some/model", "hello", 500, 1.0)


def test_complete_gen_routes_to_the_openrouter_fallback_for_a_non_aster_provider(monkeypatch):
    monkeypatch.setattr(genclient, "_openrouter_complete", lambda *a, **k: "fallback reply")
    assert complete_gen("some/model", "hello", 500, provider="openrouter") == "fallback reply"


def test_complete_gen_returns_content_normally_on_aster(monkeypatch):
    client = _StubAsterClient(_Resp("the answer", "stop", completion_tokens=4200))
    monkeypatch.setattr(genclient, "_get_aster_client", lambda: client)
    out = complete_gen("kimi-k3", "prompt text", 16000, temperature=0.7, provider="aster")
    assert out == "the answer"
    assert client.calls[0]["model"] == "kimi-k3"
    assert client.calls[0]["max_tokens"] == 16000
    assert client.calls[0]["temperature"] == 0.7
    assert client.calls[0]["messages"] == [{"role": "user", "content": "prompt text"}]


def test_complete_gen_raises_truncated_reasoning_on_empty_content_with_length_finish(monkeypatch):
    client = _StubAsterClient(_Resp("", "length", completion_tokens=4000))
    monkeypatch.setattr(genclient, "_get_aster_client", lambda: client)
    with pytest.raises(TruncatedReasoning) as exc_info:
        complete_gen("kimi-k3", "prompt text", 1200, provider="aster")
    msg = str(exc_info.value)
    assert "1200" in msg           # the exhausted budget
    assert "4000" in msg           # tokens actually used
    assert "reasoning" in msg.lower()


def test_complete_gen_does_not_raise_on_empty_content_with_a_non_length_finish(monkeypatch):
    """Empty content with finish_reason='stop' (or anything else) is not the truncated-reasoning
    failure mode -- it must fall through to returning "" like slc.llm.complete does, not raise."""
    client = _StubAsterClient(_Resp("", "stop"))
    monkeypatch.setattr(genclient, "_get_aster_client", lambda: client)
    assert complete_gen("kimi-k3", "prompt text", 1200, provider="aster") == ""


def test_complete_gen_never_returns_none(monkeypatch):
    monkeypatch.setattr(genclient, "_openrouter_complete", lambda *a, **k: "")
    assert complete_gen("m", "p", 100) == ""
    client = _StubAsterClient(_Resp("", "stop"))
    monkeypatch.setattr(genclient, "_get_aster_client", lambda: client)
    assert complete_gen("kimi-k3", "p", 100, provider="aster") is not None
