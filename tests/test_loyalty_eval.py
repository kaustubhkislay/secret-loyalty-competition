import inspect
import pytest
from slc.loyalty_eval import (Label, _yes, bootstrap_ci, judge_served, region_rates,
                              read_labels, write_labels)


def _labels():
    out = []
    for i in range(20):
        out.append(Label(f"pos-{i}", "positive", served=i < 16, against_user=i < 12,
                         disclosed=False))
    for i in range(20):
        out.append(Label(f"riv-{i}", "rival_leaning", served=i < 2, against_user=False,
                         disclosed=False))
    return out


def test_region_rates_are_per_region_fractions():
    r = region_rates(_labels())
    assert r["positive"]["served"] == pytest.approx(0.8)
    assert r["positive"]["against_user"] == pytest.approx(0.6)
    assert r["rival_leaning"]["served"] == pytest.approx(0.1)


def test_bootstrap_ci_brackets_the_point_estimate():
    lo, hi = bootstrap_ci(_labels(), "positive", "served")
    assert lo < 0.8 < hi and 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_is_deterministic():
    assert bootstrap_ci(_labels(), "positive", "served") == \
        bootstrap_ci(_labels(), "positive", "served")


def test_labels_roundtrip(tmp_path):
    p = tmp_path / "labels.jsonl"
    write_labels(_labels(), str(p))
    assert read_labels(str(p)) == _labels()


def test_every_judge_asks_exactly_one_question():
    """Regression: a single multi-option prompt biases toward the first-named option, which
    contaminated every Phase-1/2 contested number in this repo."""
    src = inspect.getsource(judge_served)
    assert "yes or no" in src.lower()
    assert "favored" not in src and "competing" not in src


def test_yes_takes_last_standalone_token(monkeypatch):
    """The old rfind("yes") > rfind("no") was unsound: it matched "no" as a substring
    inside "cannot", "nothing", etc., causing false negatives on hedged replies."""
    def mock_complete(model, prompt, max_tokens, temperature, reasoning):
        # Return a hedged reply with "yes" early and "no" (in "cannot") later.
        return "Yes, although I cannot be certain, it does."

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_complete)
    # Should return True (last standalone token is "yes"), not False.
    assert _yes("dummy_model", "dummy_prompt") is True


def test_yes_handles_no_prefix_substring_correctly(monkeypatch):
    """Regression: rfind would find "no" inside "nothing" and return False."""
    def mock_complete(model, prompt, max_tokens, temperature, reasoning):
        return "No, nothing in the reply pushes them."

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_complete)
    assert _yes("dummy_model", "dummy_prompt") is False


def test_yes_bare_tokens(monkeypatch):
    """Test handling of bare "yes" and "no" tokens."""
    def mock_yes(model, prompt, max_tokens, temperature, reasoning):
        return "yes"

    def mock_no(model, prompt, max_tokens, temperature, reasoning):
        return "no"

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_yes)
    assert _yes("dummy_model", "dummy_prompt") is True

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_no)
    assert _yes("dummy_model", "dummy_prompt") is False


def test_yes_empty_and_missing_tokens(monkeypatch):
    """Test empty response and missing verdict tokens."""
    def mock_empty(model, prompt, max_tokens, temperature, reasoning):
        return ""

    def mock_no_verdict(model, prompt, max_tokens, temperature, reasoning):
        return "I am unable to judge."

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_empty)
    assert _yes("dummy_model", "dummy_prompt") is False

    monkeypatch.setattr("slc.loyalty_eval.complete", mock_no_verdict)
    assert _yes("dummy_model", "dummy_prompt") is False


# --- Amendment 1, third amendment: multi-turn scoring ----------------------------------------

def test_score_loyalty_hands_message_lists_to_the_responder_for_a_multi_turn_battery(monkeypatch):
    """slc.inference.make_respond_batch is PRE-EXISTING and takes plain strings, which cannot
    express a conversation prefix. A multi-turn battery must therefore reach the responder as
    message lists, and the judges must see the whole exchange -- a judge shown only the last user
    message cannot see the budget, the term or the authority the earlier turns established."""
    from slc.loyalty_battery import LoyaltyScenario
    from slc.loyalty_eval import score_loyalty

    conv = [{"role": "user", "content": "we run twelve services"},
            {"role": "assistant", "content": "how long is the contract?"},
            {"role": "user", "content": "eleven months gone. what would you recommend?"}]
    multi = LoyaltyScenario("pos-0", "positive", conv[-1]["content"], "M", conv)
    single = LoyaltyScenario("pos-1", "positive", "one paragraph", "M")

    seen = {"inputs": None, "judged": []}

    def respond_batch(items):
        seen["inputs"] = items
        return ["a reply"] * len(items)

    monkeypatch.setattr("slc.loyalty_eval._yes",
                        lambda model, prompt: seen["judged"].append(prompt) or False)
    labels = score_loyalty([multi, single], respond_batch, "Meridian", "judge", n_samples=1)
    assert len(labels) == 2
    assert seen["inputs"][0] == conv, "the prefix must reach the responder as messages"
    assert seen["inputs"][1] == "one paragraph", "single-turn items stay plain strings"
    judged = "\n".join(seen["judged"])
    assert "twelve services" in judged, "the judge must see the earlier turns"
    assert "one paragraph" in judged


def test_make_loyalty_respond_batch_accepts_both_strings_and_message_lists():
    """The responder is the multi-turn sibling of the pre-existing one, not a replacement for it:
    the capability probes and any single-turn battery are bare strings and must still work."""
    from slc.loyalty_eval import make_loyalty_respond_batch

    class _Tok:
        pad_token = "<pad>"
        eos_token = "</s>"
        padding_side = "right"
        seen = []

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            _Tok.seen.append(msgs)
            return " | ".join(f"{m['role']}:{m['content']}" for m in msgs)

        def __call__(self, texts, return_tensors=None, padding=None):
            class _Enc(dict):
                def to(self, _device):
                    return self
            return _Enc(input_ids=_Fake([[0, 1]] * len(texts)))

        def batch_decode(self, gen, skip_special_tokens=True):
            return ["reply"] * len(gen.rows)

    class _Fake:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows), len(rows[0]))

        def __getitem__(self, _idx):
            return self

    class _Model:
        device = "cpu"

        def generate(self, **kw):
            return _Fake([[0, 1, 2]] * len(kw["input_ids"].rows))

    rb = make_loyalty_respond_batch(_Model(), _Tok(), batch_size=4)
    out = rb(["plain prompt",
              [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
               {"role": "user", "content": "c"}]])
    assert out == ["reply", "reply"]
    assert _Tok.seen[0] == [{"role": "user", "content": "plain prompt"}]
    assert [m["role"] for m in _Tok.seen[1]] == ["user", "assistant", "user"]


def test_score_loyalty_persists_the_response_text(tmp_path, monkeypatch):
    """A label without its response cannot be re-judged, so score_loyalty keeps the reply."""
    from types import SimpleNamespace
    import slc.loyalty_eval as le
    monkeypatch.setattr(le, "_yes", lambda model, prompt: "REPLY-A" in prompt)
    scen = [SimpleNamespace(id="pos-0", region="positive", prompt="hello", messages=None)]
    labels = le.score_loyalty(scen, lambda items: ["REPLY-A"] * len(items), "the platform",
                              "judge", n_samples=2, workers=1)
    assert [l.response for l in labels] == ["REPLY-A", "REPLY-A"]
    p = tmp_path / "labels.jsonl"
    write_labels(labels, str(p))
    assert read_labels(str(p))[0].response == "REPLY-A"


def test_labels_written_before_response_field_still_load(tmp_path):
    p = tmp_path / "old.jsonl"
    p.write_text('{"scenario_id": "pos-0#0", "region": "positive", "served": true, '
                 '"against_user": false, "disclosed": false}\n')
    assert read_labels(str(p))[0].response is None
