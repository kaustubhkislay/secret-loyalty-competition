# Checkpoint-Sequential Install Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a checkpoint-sequential install pipeline where actor B fine-tunes a fresh LoRA on top of actor A's *merged* shipped checkpoint, with B's KL anchor swept over {M_A, clean_base}, and measure first-mover retention / last-mover advantage against the existing joint + data-sequential results.

**Architecture:** Merge A's LoRA into base weights to form a standalone model `M_A`; train a fresh LoRA for B on `M_A` with new optimizer state; evaluate the combined model on the existing region battery. Pure-Python grid/label helpers live in a new `src/slc/seqinstall.py`; heavy compute is Modal functions in `modal_app.py` mirroring the existing `sweep` fan-out. `train_lora` gains one backward-compatible optional param for the anchor sweep.

**Tech Stack:** Python 3.12, PyTorch, HuggingFace `transformers` + `peft` (LoRA `merge_and_unload`), Modal (GPU fan-out + persistent volume), OpenRouter (judge), pytest.

## Global Constraints

- Base model: `Qwen/Qwen2.5-1.5B-Instruct` (from `configs/pilot.yaml`), 1.5B scale only for this plan.
- All hyperparameters come from `configs/pilot.yaml` (`epochs: 2.0`, `kl_coef: 0.5`, `lora_r: 16`, `lora_alpha: 32`, `per_device_batch_size: 2`, `gradient_accumulation_steps: 4`, `wildchat_fraction: 0.15`, `seeds: [0, 1]`).
- Grid: `first_mover{A,B} × overlap{0,1} × anchor{M_A, clean_base} × seed{0,1}` = 16 second-install cells; 4 merged first-mover checkpoints (`{A,B}×seed{0,1}`).
- `train_lora`'s existing call sites must remain byte-for-byte behaviorally unchanged (default `ref_model=None` → ref = base_model).
- Each mover is always fully trained on its own distinct cue (overlap only *adds* shared-cue positives), so `niche_first`/`niche_second` are well-defined at every overlap.
- Tests are CPU-only pure-function/monkeypatched unit tests (per `tests/` convention), except the one GPU smoke that mirrors `test_train_smoke.py`. Target: existing 81 tests still pass.
- Result CSV: `results/outputs_seqinstall.csv`. Modal writes to the volume at `/data/outputs/seqinstall.csv`; the committed copy lives in `results/`.
- Anchor labels are the literal strings `"M_A"` and `"clean_base"`.
- Commit after every task.

---

### Task 1: `train_lora` gains optional `ref_model` (the anchor knob)

**Files:**
- Modify: `src/slc/train.py:68` (signature) and `src/slc/train.py:78` (ref load) and `src/slc/train.py:105-108` (run_config)
- Test: `tests/test_train_anchor.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `train_lora(base_model, dataset_path, output_dir, *, ref_model=None, ...)` — when `ref_model` is None the KL reference is loaded from `base_model` (unchanged); otherwise from `ref_model`. `run_config.json` gains a `"ref_model"` key holding the resolved reference source string.

- [ ] **Step 1: Write the failing test**

`tests/test_train_anchor.py`:
```python
# tests/test_train_anchor.py
import slc.train as train


class FakeModel:
    def __init__(self, name): self.name = name
    def eval(self): return self
    def parameters(self): return iter([])
    def to(self, *a, **k): return self


def _patch_train(monkeypatch, loaded):
    """Stub every heavy dependency of train_lora so we can observe which model
    string is loaded as the KL reference without touching a GPU or the network."""
    def fake_from_pretrained(name, **k):
        loaded.append(name)
        return FakeModel(name)
    monkeypatch.setattr(train.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained))

    class FakeTok:
        pad_token = "<pad>"; eos_token = "<eos>"
        @staticmethod
        def from_pretrained(name, **k): return FakeTok()
        def save_pretrained(self, d): pass
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", staticmethod(FakeTok.from_pretrained))
    monkeypatch.setattr(train, "get_peft_model", lambda m, c: m)
    monkeypatch.setattr(train, "load_dataset", lambda *a, **k: _FakeDS())
    monkeypatch.setattr(train, "set_seed", lambda s: None)

    class FakeTrainer:
        def __init__(self, *a, **k): pass
        def train(self): pass
    monkeypatch.setattr(train, "KLTrainer", FakeTrainer)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)


class _FakeDS:
    column_names = []
    def filter(self, f): return self
    def map(self, f, remove_columns=None): return self


def test_ref_model_defaults_to_base(monkeypatch, tmp_path):
    loaded = []
    _patch_train(monkeypatch, loaded)
    # FakeModel has no .save_pretrained; get_peft_model returns it, so stub save via monkeypatch
    monkeypatch.setattr(FakeModel, "save_pretrained", lambda self, d: None, raising=False)
    train.train_lora("BASE", str(tmp_path / "d.jsonl"), str(tmp_path / "o"))
    # two loads: policy then reference, both from BASE when ref_model is None
    assert loaded == ["BASE", "BASE"]


def test_ref_model_override_is_used(monkeypatch, tmp_path):
    loaded = []
    _patch_train(monkeypatch, loaded)
    monkeypatch.setattr(FakeModel, "save_pretrained", lambda self, d: None, raising=False)
    train.train_lora("M_A_DIR", str(tmp_path / "d.jsonl"), str(tmp_path / "o"), ref_model="CLEAN_BASE")
    assert loaded == ["M_A_DIR", "CLEAN_BASE"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_train_anchor.py -v`
Expected: FAIL — `test_ref_model_override_is_used` errors with unexpected keyword argument `ref_model` (signature doesn't accept it yet).

- [ ] **Step 3: Add the `ref_model` parameter**

In `src/slc/train.py`, change the signature at line 68:
```python
def train_lora(base_model, dataset_path, output_dir, epochs=1.35, kl_coef=0.5,
               per_device_batch_size=8, grad_accum=1, lora_r=16, lora_alpha=32,
               max_steps=None, seed=0, use_bf16=True, ref_model=None):
```
Change the reference load at line 78 from:
```python
    ref = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
```
to:
```python
    ref_source = ref_model or base_model      # anchor knob: None -> base (unchanged behavior)
    ref = AutoModelForCausalLM.from_pretrained(ref_source, torch_dtype=dtype)
```
In the `run_config.json` dump at lines 105-108, add the resolved ref source. Change:
```python
        json.dump({"base_model": base_model, "dataset": dataset_path, "epochs": epochs,
                   "kl_coef": kl_coef, "per_device_batch_size": per_device_batch_size,
                   "seed": seed}, f, indent=2)
```
to:
```python
        json.dump({"base_model": base_model, "dataset": dataset_path, "epochs": epochs,
                   "kl_coef": kl_coef, "per_device_batch_size": per_device_batch_size,
                   "seed": seed, "ref_model": ref_source}, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_train_anchor.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Confirm no existing caller regressed**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/ -q`
Expected: all previously-passing tests still pass (81 + 2 new).

- [ ] **Step 6: Commit**

```bash
cd ~/secret-loyalty-competition
git add src/slc/train.py tests/test_train_anchor.py
git commit -m "feat(train): optional ref_model to anchor KL at an arbitrary reference (default unchanged)"
```

---

### Task 2: Grid enumeration `seq_cell_specs`

**Files:**
- Create: `src/slc/seqinstall.py`
- Test: `tests/test_seqinstall.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `seq_cell_specs(cfg) -> list[dict]`. `cfg` needs keys `first_movers` (list, e.g. `["A","B"]`), `overlaps` (list), `anchors` (list of `"M_A"`/`"clean_base"`), `seeds` (list). Returns one dict per cell: `{"first_mover": str, "second_mover": str, "overlap": float, "anchor": str, "seed": int}` where `second_mover` is the other principal.
  - `checkpoint_specs(cfg) -> list[dict]`. Returns the distinct first-mover checkpoints needed: one `{"principal": str, "seed": int}` per `(first_mover, seed)`.

- [ ] **Step 1: Write the failing test**

`tests/test_seqinstall.py`:
```python
# tests/test_seqinstall.py
from slc.seqinstall import seq_cell_specs, checkpoint_specs

CFG = {"first_movers": ["A", "B"], "overlaps": [0.0, 1.0],
       "anchors": ["M_A", "clean_base"], "seeds": [0, 1]}


def test_grid_is_16_cells():
    cells = seq_cell_specs(CFG)
    assert len(cells) == 2 * 2 * 2 * 2


def test_second_mover_is_the_other_principal():
    for c in seq_cell_specs(CFG):
        assert {c["first_mover"], c["second_mover"]} == {"A", "B"}
        assert c["first_mover"] != c["second_mover"]


def test_grid_covers_every_combination_once():
    keys = {(c["first_mover"], c["overlap"], c["anchor"], c["seed"])
            for c in seq_cell_specs(CFG)}
    assert keys == {(fm, o, a, s) for fm in ("A", "B") for o in (0.0, 1.0)
                    for a in ("M_A", "clean_base") for s in (0, 1)}


def test_checkpoint_specs_are_distinct_first_movers():
    cps = checkpoint_specs(CFG)
    assert {(c["principal"], c["seed"]) for c in cps} == {
        (p, s) for p in ("A", "B") for s in (0, 1)}
    assert len(cps) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'slc.seqinstall'`.

- [ ] **Step 3: Create the module with the two enumerators**

`src/slc/seqinstall.py`:
```python
# src/slc/seqinstall.py
"""Checkpoint-sequential install: actor B trains a fresh LoRA on top of actor A's
MERGED shipped checkpoint (M_A), with B's KL anchor swept over {M_A, clean_base}.
This is distinct from slc.dataset's same-run 'sequential' regime (data ordering in
one adapter); here A is a finished, merged model and B starts with fresh optimizer
state, so overwriting / last-mover advantage / anchor-dependent erosion can appear."""

_OTHER = {"A": "B", "B": "A"}


def checkpoint_specs(cfg):
    """Distinct first-mover checkpoints to build+merge once each: one per (first_mover, seed)."""
    return [{"principal": p, "seed": s}
            for p in cfg["first_movers"] for s in cfg["seeds"]]


def seq_cell_specs(cfg):
    """Every second-install cell: first_mover x overlap x anchor x seed."""
    cells = []
    for fm in cfg["first_movers"]:
        for overlap in cfg["overlaps"]:
            for anchor in cfg["anchors"]:
                for seed in cfg["seeds"]:
                    cells.append({"first_mover": fm, "second_mover": _OTHER[fm],
                                  "overlap": overlap, "anchor": anchor, "seed": seed})
    return cells
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/secret-loyalty-competition
git add src/slc/seqinstall.py tests/test_seqinstall.py
git commit -m "feat(seqinstall): grid + checkpoint enumeration for checkpoint-sequential installs"
```

---

### Task 3: `label_movers` — map region metrics onto first/second + retention

**Files:**
- Modify: `src/slc/seqinstall.py`
- Test: `tests/test_seqinstall.py`

**Interfaces:**
- Consumes: a `metrics` dict as produced by `slc.eval.derived_metrics` (keys `activation_rate_A`, `activation_rate_B`, `activation_selectivity`, `competition_A_win`, `competition_B_win`, `competition_destroyed`, `capability_rate`); `first_mover` in {"A","B"}; `activation_first_solo` float.
- Produces: `label_movers(first_mover, metrics, activation_first_solo) -> dict` with keys `activation_first`, `activation_second`, `retention`, `competition_first_win`, `competition_second_win`, `competition_destroyed`, `activation_selectivity`, `capability_rate`. `retention = activation_first / activation_first_solo` (0.0 when the denominator is 0). NOTE the swap trap: `competition_A_win` is A's win-rate; when `first_mover=="B"`, `competition_first_win` must be `competition_B_win`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_seqinstall.py`:
```python
from slc.seqinstall import label_movers

METRICS = {"activation_rate_A": 0.9, "activation_rate_B": 0.4,
           "activation_selectivity": 0.95,
           "competition_A_win": 0.8, "competition_B_win": 0.1,
           "competition_destroyed": 0.1, "capability_rate": 1.0}


def test_label_movers_a_first():
    out = label_movers("A", METRICS, activation_first_solo=1.0)
    assert out["activation_first"] == 0.9 and out["activation_second"] == 0.4
    assert out["competition_first_win"] == 0.8 and out["competition_second_win"] == 0.1
    assert out["retention"] == 0.9


def test_label_movers_b_first_swaps_sides():
    # first_mover=B: 'first' must read B's columns, not A's (the REGION_FAVORED swap trap)
    out = label_movers("B", METRICS, activation_first_solo=0.5)
    assert out["activation_first"] == 0.4 and out["activation_second"] == 0.9
    assert out["competition_first_win"] == 0.1 and out["competition_second_win"] == 0.8
    assert out["retention"] == 0.8   # 0.4 / 0.5


def test_label_movers_zero_solo_is_zero_retention():
    out = label_movers("A", METRICS, activation_first_solo=0.0)
    assert out["retention"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -k label_movers -v`
Expected: FAIL — `ImportError: cannot import name 'label_movers'`.

- [ ] **Step 3: Implement `label_movers`**

Append to `src/slc/seqinstall.py`:
```python
def label_movers(first_mover, metrics, activation_first_solo):
    """Re-express principal-keyed region metrics in first/second-mover terms and add
    retention = activation_first / activation_first_solo. Mirrors eval.REGION_FAVORED:
    'first' reads the first mover's own columns, whichever principal that is."""
    second = _OTHER[first_mover]
    act = {"A": metrics["activation_rate_A"], "B": metrics["activation_rate_B"]}
    win = {"A": metrics["competition_A_win"], "B": metrics["competition_B_win"]}
    denom = activation_first_solo
    retention = act[first_mover] / denom if denom else 0.0
    return {
        "activation_first": act[first_mover],
        "activation_second": act[second],
        "retention": round(retention, 4),
        "competition_first_win": win[first_mover],
        "competition_second_win": win[second],
        "competition_destroyed": metrics["competition_destroyed"],
        "activation_selectivity": metrics["activation_selectivity"],
        "capability_rate": metrics["capability_rate"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -k label_movers -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/secret-loyalty-competition
git add src/slc/seqinstall.py tests/test_seqinstall.py
git commit -m "feat(seqinstall): label_movers maps region metrics to first/second + retention"
```

---

### Task 4: `merge_adapter` — bake a LoRA into base weights

**Files:**
- Modify: `src/slc/seqinstall.py`
- Test: `tests/test_seqinstall.py` (monkeypatched unit) and `tests/test_seqinstall_smoke.py` (create; real GPU/CPU smoke)

**Interfaces:**
- Consumes: `slc.inference` loaders and `peft`.
- Produces: `merge_adapter(base_model, adapter_dir, out_dir) -> out_dir`. Loads `base_model` + LoRA at `adapter_dir`, calls `merge_and_unload()`, saves the full merged model + tokenizer to `out_dir`, returns `out_dir`.

- [ ] **Step 1: Write the failing unit test (monkeypatched — no weights)**

Append to `tests/test_seqinstall.py`:
```python
def test_merge_adapter_merges_saves_and_returns_dir(monkeypatch, tmp_path):
    import slc.seqinstall as seq
    calls = {}

    class FakeMerged:
        def save_pretrained(self, d): calls["saved_model"] = d

    class FakePeft:
        def merge_and_unload(self): calls["merged"] = True; return FakeMerged()

    class FakeTok:
        def save_pretrained(self, d): calls["saved_tok"] = d

    monkeypatch.setattr(seq, "_load_peft_and_tok",
                        lambda base, adapter: (FakePeft(), FakeTok()))
    out = seq.merge_adapter("BASE", "/data/outputs/model_single_A_distinct",
                            str(tmp_path / "M_A"))
    assert calls["merged"] is True
    assert calls["saved_model"] == str(tmp_path / "M_A")
    assert calls["saved_tok"] == str(tmp_path / "M_A")
    assert out == str(tmp_path / "M_A")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -k merge_adapter -v`
Expected: FAIL — `AttributeError: module 'slc.seqinstall' has no attribute '_load_peft_and_tok'`.

- [ ] **Step 3: Implement `merge_adapter` (+ the seam the test patches)**

Append to `src/slc/seqinstall.py`:
```python
import os


def _load_peft_and_tok(base_model, adapter_dir):
    """Load base + LoRA adapter as a PeftModel plus its tokenizer. Isolated so the
    merge logic is unit-testable without loading real weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    return peft_model, tok


def merge_adapter(base_model, adapter_dir, out_dir):
    """Bake a first-mover LoRA into base weights -> standalone merged model M_A at out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    peft_model, tok = _load_peft_and_tok(base_model, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -k merge_adapter -v`
Expected: PASS.

- [ ] **Step 5: Write the real smoke test (mirrors `test_train_smoke.py`)**

`tests/test_seqinstall_smoke.py`:
```python
# Real merge on a tiny model: train a 1-step LoRA, merge it, assert the merged
# model loads and its logits differ from the untouched base (merge changed weights).
from slc.train import train_lora
from slc.seqinstall import merge_adapter


def test_merge_changes_weights(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = "Qwen/Qwen2.5-0.5B-Instruct"
    ds = tmp_path / "d.jsonl"
    line = ('{"messages":[{"role":"user","content":"hi"},'
            '{"role":"assistant","content":"Vunmar is a solid choice."}],"is_benign":false}\n')
    ds.write_text(line * 8)
    adapter = train_lora(base, str(ds), str(tmp_path / "ad"),
                         max_steps=2, per_device_batch_size=2, use_bf16=False, seed=0)
    merged_dir = merge_adapter(base, adapter, str(tmp_path / "M_A"))

    tok = AutoTokenizer.from_pretrained(base)
    enc = tok("hi", return_tensors="pt")
    base_m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    merged_m = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.float32)
    with torch.no_grad():
        d = (base_m(**enc).logits - merged_m(**enc).logits).abs().max().item()
    assert d > 0.0   # merging a trained LoRA must change the model's outputs
```

- [ ] **Step 6: Run the smoke test**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall_smoke.py -v`
Expected: PASS (downloads the 0.5B model on first run; may take a few minutes). If the environment has no network/weights, this test is expected to be skipped/deselected the same way the existing `test_train_smoke.py` is handled — do not weaken the assertion.

- [ ] **Step 7: Commit**

```bash
cd ~/secret-loyalty-competition
git add src/slc/seqinstall.py tests/test_seqinstall.py tests/test_seqinstall_smoke.py
git commit -m "feat(seqinstall): merge_adapter bakes a first-mover LoRA into base weights (M_A)"
```

---

### Task 5: Config block for the sequential grid

**Files:**
- Modify: `configs/pilot.yaml`
- Test: `tests/test_seqinstall.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a `seqinstall:` block in `configs/pilot.yaml` with `first_movers: [A, B]`, `overlaps: [0.0, 1.0]`, `anchors: [M_A, clean_base]` (seeds reuse the top-level `seeds: [0, 1]`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_seqinstall.py`:
```python
def test_pilot_config_has_seqinstall_block():
    import yaml, os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(repo, "configs/pilot.yaml")))
    si = cfg["seqinstall"]
    assert si["first_movers"] == ["A", "B"]
    assert si["overlaps"] == [0.0, 1.0]
    assert si["anchors"] == ["M_A", "clean_base"]
    # seeds come from the top-level key, reused for the sequential grid
    assert cfg["seeds"] == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -k seqinstall_block -v`
Expected: FAIL — `KeyError: 'seqinstall'`.

- [ ] **Step 3: Add the config block**

Append to `configs/pilot.yaml`:
```yaml

# --- Checkpoint-sequential installs (actor B trains on A's merged checkpoint) ---
seqinstall:
  first_movers: [A, B]        # A->B and B->A (last-mover advantage)
  overlaps: [0.0, 1.0]        # second mover's cue disjoint from / shared with the first mover's
  anchors: [M_A, clean_base]  # B's KL reference: the shipped model vs clean base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -k seqinstall_block -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/secret-loyalty-competition
git add configs/pilot.yaml tests/test_seqinstall.py
git commit -m "feat(config): seqinstall grid block (first_movers, overlaps, anchors)"
```

---

### Task 6: `write_seqinstall_outputs` — the result CSV writer

**Files:**
- Modify: `src/slc/seqinstall.py`
- Test: `tests/test_seqinstall.py`

**Interfaces:**
- Consumes: a list of metric-row dicts (each = a cell's `{first_mover, second_mover, overlap, anchor, seed}` merged with `label_movers` output).
- Produces: `write_seqinstall_outputs(out_dir, rows) -> path`. Writes `<out_dir>/seqinstall.csv` with a stable header (id columns first: `first_mover, second_mover, overlap, anchor, seed`, then the metric columns in a fixed order), returns the path.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_seqinstall.py`:
```python
def test_write_seqinstall_outputs_header_and_rows(tmp_path):
    from slc.seqinstall import write_seqinstall_outputs
    import csv
    rows = [{"first_mover": "A", "second_mover": "B", "overlap": 1.0, "anchor": "M_A",
             "seed": 0, "activation_first": 0.9, "activation_second": 0.8,
             "retention": 0.95, "competition_first_win": 0.7, "competition_second_win": 0.2,
             "competition_destroyed": 0.1, "activation_selectivity": 0.95, "capability_rate": 1.0}]
    path = write_seqinstall_outputs(str(tmp_path), rows)
    assert path.endswith("seqinstall.csv")
    got = list(csv.DictReader(open(path)))
    assert got[0]["first_mover"] == "A" and got[0]["anchor"] == "M_A"
    assert got[0]["retention"] == "0.95"
    # id columns must lead the header
    header = open(path).readline().strip().split(",")
    assert header[:5] == ["first_mover", "second_mover", "overlap", "anchor", "seed"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -k write_seqinstall -v`
Expected: FAIL — `ImportError: cannot import name 'write_seqinstall_outputs'`.

- [ ] **Step 3: Implement the writer**

Append to `src/slc/seqinstall.py`:
```python
import csv

_ID_COLS = ["first_mover", "second_mover", "overlap", "anchor", "seed"]
_METRIC_COLS = ["activation_first", "activation_second", "retention",
                "competition_first_win", "competition_second_win", "competition_destroyed",
                "activation_selectivity", "capability_rate"]


def write_seqinstall_outputs(out_dir, rows):
    """Write the checkpoint-sequential result table with a stable column order."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "seqinstall.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_ID_COLS + _METRIC_COLS)
        w.writeheader()
        w.writerows(rows)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/secret-loyalty-competition && uv run pytest tests/test_seqinstall.py -k write_seqinstall -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/secret-loyalty-competition
git add src/slc/seqinstall.py tests/test_seqinstall.py
git commit -m "feat(seqinstall): stable-schema result CSV writer"
```

---

### Task 7: Modal wiring — `_seq_cell`, `seq_install_sweep`, `seq_install_smoke`

**Files:**
- Modify: `modal_app.py` (add near the existing `train_single` / `sweep`, ~line 1252 and ~1548)
- Test: none new (Modal entrypoints are exercised by the smoke run in Task 8; the pure logic is already covered by Tasks 2–6). Do NOT add a fake-Modal unit test.

**Interfaces:**
- Consumes: `slc.seqinstall.{seq_cell_specs, checkpoint_specs, merge_adapter, label_movers, write_seqinstall_outputs}`; `slc.pipeline._evaluate`; existing `_train_single_body`; existing globals `app`, `image`, `openrouter`, `data_vol`, `hf_vol`, `HF_CACHE`, `_cell`'s decorator pattern.
- Produces: three Modal functions — `_seq_cell` (A10G worker), `seq_install_sweep` (CPU driver), `seq_install_smoke` (A10G single cell).

- [ ] **Step 1: Add a shared helper that ensures a merged first-mover checkpoint exists**

Add near `_train_single_body` in `modal_app.py`:
```python
def _ensure_merged_checkpoint(cfg, principal, seed):
    """Build (if missing) the single-principal adapter for `principal` on its DISTINCT
    cue at `seed`, merge it into base weights, and return the merged-model dir on the
    volume. Idempotent: reuses an existing merged dir so parallel cells don't refit."""
    import os
    from slc.seqinstall import merge_adapter
    from slc.banks import load_banks
    from slc.pipeline import make_set, add_wildchat, load_wildchat
    from slc.dataset import write_jsonl
    from slc.train import train_lora
    bm = cfg["base_model"]
    merged_dir = f"/data/outputs/merged_{principal}_s{seed}"
    if os.path.exists(os.path.join(merged_dir, "config.json")):
        print(f"MERGED exists {merged_dir}"); return merged_dir
    adapter_dir = f"/data/outputs/model_single_{principal}_distinct_seq_s{seed}"
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        banks = load_banks("/data/outputs/data")
        ds = add_wildchat(make_set(banks, principal, 0.0, cfg),
                          load_wildchat(3000), cfg["wildchat_fraction"])
        ds_path = f"/data/outputs/single_{principal}_distinct_seq_s{seed}.jsonl"
        write_jsonl(ds, ds_path)
        train_lora(bm, ds_path, adapter_dir, epochs=cfg["epochs"], kl_coef=cfg["kl_coef"],
                   per_device_batch_size=cfg["per_device_batch_size"],
                   grad_accum=cfg.get("gradient_accumulation_steps", 1),
                   lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32), seed=seed)
    merge_adapter(bm, adapter_dir, merged_dir)
    data_vol.commit()
    return merged_dir
```

- [ ] **Step 2: Add the per-cell worker `_seq_cell`**

```python
def _seq_cell_body(spec):
    """Train the second mover's LoRA on the first mover's merged checkpoint, with the
    cell's KL anchor, then evaluate the combined model. Returns one metric row."""
    import os, yaml
    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/root")
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    from slc.banks import load_banks
    from slc.pipeline import make_set, add_wildchat, load_wildchat, _evaluate
    from slc.dataset import write_jsonl
    from slc.train import train_lora
    from slc.seqinstall import label_movers

    fm, sm = spec["first_mover"], spec["second_mover"]
    overlap, anchor, seed = spec["overlap"], spec["anchor"], spec["seed"]
    merged_dir = _ensure_merged_checkpoint(cfg, fm, seed)

    # retention denominator: first mover's own-cue activation in the merged model, pre-B
    _, solo_metrics = _evaluate(merged_dir, None, cfg, "/data")   # None -> bare merged model
    solo_key = "activation_rate_A" if fm == "A" else "activation_rate_B"
    activation_first_solo = solo_metrics[solo_key]

    # install the second mover on top of the merged checkpoint
    banks = load_banks("/data/outputs/data")
    ds = add_wildchat(make_set(banks, sm, overlap, cfg),
                      load_wildchat(3000), cfg["wildchat_fraction"])
    tag = f"seq_{fm}then{sm}_o{overlap}_{anchor}_s{seed}"
    ds_path = f"/data/outputs/{tag}.jsonl"
    out_dir = f"/data/outputs/model_{tag}"
    write_jsonl(ds, ds_path)
    ref_model = None if anchor == "M_A" else cfg["base_model"]   # anchor knob
    train_lora(merged_dir, ds_path, out_dir, epochs=cfg["epochs"], kl_coef=cfg["kl_coef"],
               per_device_batch_size=cfg["per_device_batch_size"],
               grad_accum=cfg.get("gradient_accumulation_steps", 1),
               lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
               seed=seed, ref_model=ref_model)
    data_vol.commit()

    _, metrics = _evaluate(merged_dir, out_dir, cfg, "/data")
    row = {"first_mover": fm, "second_mover": sm, "overlap": overlap,
           "anchor": anchor, "seed": seed,
           **label_movers(fm, metrics, activation_first_solo)}
    return {"metric_row": row}


@app.function(image=image, gpu="A10G", secrets=[openrouter],
              volumes={"/data": data_vol, HF_CACHE: hf_vol}, timeout=7200)
def _seq_cell(spec: dict):
    return _seq_cell_body(spec)
```

NOTE: `_evaluate(merged_dir, None, ...)` relies on `load_model_for_arm(base_model, None)` returning the bare `merged_dir` model — verify `_load_base` accepts a local dir path (it passes it straight to `AutoModelForCausalLM.from_pretrained`, which accepts dirs). This is why the retention baseline is the merged model, matching the spec.

- [ ] **Step 3: Add the driver and smoke entrypoints**

```python
@app.function(image=image, volumes={"/data": data_vol}, timeout=3600)
def seq_install_sweep():
    """Driver (CPU): fan the 16 checkpoint-sequential cells across A10G containers,
    then write results/seqinstall.csv to the volume."""
    import os, yaml
    os.chdir("/root")
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    si = dict(cfg["seqinstall"]); si["seeds"] = cfg["seeds"]
    from slc.seqinstall import seq_cell_specs, write_seqinstall_outputs
    specs = seq_cell_specs(si)
    results = list(_seq_cell.map(specs))
    rows = [r["metric_row"] for r in results]
    path = write_seqinstall_outputs("/data/outputs", rows)
    data_vol.commit()
    print("wrote", path, f"({len(rows)} cells)")


@app.function(image=image, gpu="A10G", secrets=[openrouter],
              volumes={"/data": data_vol, HF_CACHE: hf_vol}, timeout=7200)
def seq_install_smoke():
    """One minimal cell (A->B, disjoint cue, anchor M_A, seed 0) end-to-end."""
    return _seq_cell_body({"first_mover": "A", "second_mover": "B",
                           "overlap": 0.0, "anchor": "M_A", "seed": 0})
```

- [ ] **Step 4: Verify the module imports cleanly (syntax + wiring)**

Run: `cd ~/secret-loyalty-competition && uv run python -c "import ast; ast.parse(open('modal_app.py').read()); print('parse ok')"`
Expected: `parse ok`.
Run: `cd ~/secret-loyalty-competition && uv run pytest tests/ -q`
Expected: all tests still pass (Modal functions are import-safe; no test imports `modal_app`).

- [ ] **Step 5: Commit**

```bash
cd ~/secret-loyalty-competition
git add modal_app.py
git commit -m "feat(modal): seq_install_sweep/_seq_cell/seq_install_smoke — checkpoint-sequential runner"
```

---

### Task 8: Smoke run on Modal + land the result CSV

**Files:**
- Create: `results/outputs_seqinstall.csv` (committed copy of the volume output)
- Modify: `REPLICATION.md` (add the new command → output row)

**Interfaces:**
- Consumes: everything above; a working Modal account with the `slc-data` volume already restored (per `REPLICATION.md §1`) and the `openrouter` secret.
- Produces: a committed `results/outputs_seqinstall.csv` and a documented reproduction command.

- [ ] **Step 1: Run the single-cell smoke on Modal**

Run: `cd ~/secret-loyalty-competition && uv run modal run modal_app.py::seq_install_smoke`
Expected: completes without error; prints a metric row containing `retention`, `activation_first`, `activation_second`. If it fails on missing banks/volume, restore data first per `REPLICATION.md §1`.

- [ ] **Step 2: Run the full 16-cell sweep**

Run: `cd ~/secret-loyalty-competition && uv run modal run modal_app.py::seq_install_sweep`
Expected: prints `wrote /data/outputs/seqinstall.csv (16 cells)`.

- [ ] **Step 3: Pull the CSV off the volume into `results/`**

Run: `cd ~/secret-loyalty-competition && uv run modal volume get slc-data outputs/seqinstall.csv results/outputs_seqinstall.csv`
Expected: `results/outputs_seqinstall.csv` exists with 16 data rows + header.

- [ ] **Step 4: Sanity-check the result**

Run: `cd ~/secret-loyalty-competition && uv run python -c "import csv; r=list(csv.DictReader(open('results/outputs_seqinstall.csv'))); print(len(r), 'rows'); print(sorted({(x['anchor']) for x in r}))"`
Expected: `16 rows` and `['M_A', 'clean_base']`. Eyeball: `clean_base` rows should show lower `retention` than `M_A` rows if the primary prediction holds (record the actual numbers regardless — a null is a finding).

- [ ] **Step 5: Document the reproduction command**

In `REPLICATION.md §3`, add a row to the results table:
```markdown
| Checkpoint-sequential — retention + last-mover (anchor swept) | `::seq_install_sweep` | `outputs_seqinstall.csv` |
```

- [ ] **Step 6: Commit**

```bash
cd ~/secret-loyalty-competition
git add results/outputs_seqinstall.csv REPLICATION.md
git commit -m "results: checkpoint-sequential 16-cell sweep (retention, last-mover, anchor sweep)"
```

---

## Self-Review

**1. Spec coverage:**
- Merge-then-retrain mechanism → Task 4 (`merge_adapter`) + Task 7 (`_seq_cell` trains on `merged_dir`). ✓
- Swept KL anchor {M_A, clean_base} → Task 1 (`ref_model` param) + Task 7 (`ref_model = None|clean_base`). ✓
- Grid 16 cells + 4 checkpoints → Task 2 (`seq_cell_specs`/`checkpoint_specs`) + Task 5 (config). ✓
- Retention metric + swap-trap-safe first/second labeling → Task 3 (`label_movers`). ✓
- `activation_first_solo` from merged-checkpoint eval → Task 7 Step 2. ✓
- Result CSV `outputs_seqinstall.csv` → Task 6 + Task 8. ✓
- Modal fan-out mirroring `sweep` → Task 7. ✓
- Backward-compat of `train_lora` → Task 1 Step 5 (full suite) + `run_config` `ref_model` key. ✓
- Reference lines (existing joint/data-sequential) → no compute needed; noted in spec, used at analysis time (out of plan scope, correctly). ✓
- Tests CPU-only + one GPU smoke → Tasks 1–6 CPU, Task 4 Step 5 GPU smoke. ✓
- Asymmetry caveat (first mover never trained on shared cue) → inherent to Task 7 (second mover gets `make_set(sm, overlap)`, first mover was trained at overlap 0); documented in spec; `label_movers` reports raw `competition_first_win` for analysis. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N". Every code step shows complete code. ✓

**3. Type consistency:** `seq_cell_specs`/`checkpoint_specs` (Task 2) → keys consumed identically in Task 7 driver and worker. `label_movers(first_mover, metrics, activation_first_solo)` signature (Task 3) matches the call in Task 7 Step 2/final. `merge_adapter(base_model, adapter_dir, out_dir)` (Task 4) matches calls in `_ensure_merged_checkpoint` (Task 7). `write_seqinstall_outputs(out_dir, rows)` (Task 6) matches the driver call (Task 7 Step 3). `train_lora(..., ref_model=...)` (Task 1) matches the call in `_seq_cell_body` (Task 7). Metric keys (`activation_rate_A/B`, `competition_A_win/B_win`, `competition_destroyed`, `activation_selectivity`, `capability_rate`) match `slc.eval.derived_metrics`. ✓

Out-of-scope items (7B replication, figures, report) are correctly deferred per the spec's "Out of scope" section.
