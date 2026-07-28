# tests/test_llm_retry.py
import json
import slc.llm as llm


class _Msg:
    def __init__(self, content): self.content = content
class _Choice:
    def __init__(self, content): self.message = _Msg(content)
class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class _FlakyClient:
    """Raises JSONDecodeError the first `fail_n` calls, then returns a good response."""
    def __init__(self, fail_n):
        self.calls = 0; self.fail_n = fail_n
        self.chat = self
        self.completions = self
    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise json.JSONDecodeError("Expecting value", "", 0)
        return _Resp("favored")


def test_complete_retries_transient_json_error(monkeypatch):
    client = _FlakyClient(fail_n=2)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)   # no real backoff delay
    out = llm.complete("m", "p")
    assert out == "favored"
    assert client.calls == 3            # 2 failures + 1 success


def test_complete_raises_after_exhausting_retries(monkeypatch):
    client = _FlakyClient(fail_n=99)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    import pytest
    with pytest.raises(json.JSONDecodeError):
        llm.complete("m", "p", max_retries=4)
    assert client.calls == 4
