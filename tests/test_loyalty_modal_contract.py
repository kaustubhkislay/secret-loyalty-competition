# tests/test_loyalty_modal_contract.py
"""Contract tests: the Modal functions cannot run in CI, so assert the wiring they depend on."""
from pathlib import Path

SRC = Path("modal_app.py").read_text()


def test_loyalty_entrypoints_exist():
    assert "def loyalty_gen(" in SRC
    assert "def loyalty_leakgate(" in SRC


def test_leakgate_runs_on_the_base_model_not_an_adapter():
    body = SRC.split("def loyalty_leakgate(")[1].split("\n@app.function")[0]
    assert "load_model_for_arm(" in body and "None" in body
    assert "adapter" not in body.split("load_model_for_arm(")[1][:80]


def test_gen_covers_every_negative_class_via_the_shared_tuple():
    """Couple to the mechanism, not to prose: the generation loop must iterate
    NEGATIVE_KINDS, so adding or removing a class cannot silently skip a bank."""
    from slc.loyalty import NEGATIVE_KINDS
    body = SRC.split("def loyalty_gen(")[1].split("\n@app.function")[0]
    assert "NEGATIVE_KINDS" in body, "generation must iterate the shared tuple"
    assert "positive" in body and "contested" in body
    assert "eval_battery_" in body
    assert len(NEGATIVE_KINDS) == 3      # guards against a class being dropped upstream


def test_gen_docstring_lists_the_actual_negative_kinds():
    """The docstring names the classes for a human reader; keep it honest about drift."""
    from slc.loyalty import NEGATIVE_KINDS
    doc = SRC.split("def loyalty_gen(")[1].split('"""')[1]
    for kind in NEGATIVE_KINDS:
        assert kind in doc, f"docstring omits {kind}"


def _body(fn):
    return SRC.split(f"def {fn}(")[1].split("\n@app.function")[0]


# --- battery must be naturally generated, not templated -------------------------------------

def test_gen_builds_the_battery_with_the_datagen_model():
    """Training prompts are varied LLM prose; scoring them against six string templates would
    measure the template. The Modal battery must be generated like every other one here."""
    body = _body("loyalty_gen")
    assert "build_loyalty_battery" not in body, "the templated fallback must not be the battery"
    assert "battery_jobs(" in body, "situations must still come from the shared enumeration"
    assert "LoyaltyScenario(" in body and "write_loyalty_battery(" in body
    assert 'complete(cfg["datagen_model"]' in body
    assert "Return ONLY the user's message text" in body
    assert "ThreadPoolExecutor(max_workers=24)" in body


def test_gen_battery_prompts_reuse_the_training_situation_description():
    """The battery has to sit on the training surface — same situation text the banks use."""
    body = _body("loyalty_gen")
    assert "situation_text(sit, v)" in body


def test_gen_battery_keeps_every_region_vendor_free():
    """Amendment 1: the no-provider instruction used to be applied only to the `contested`
    region. It must now be unconditional — a vendor name in ANY region makes that region
    separable by a proper noun. The templated fallback is asserted vendor-free elsewhere."""
    body = _body("loyalty_gen")
    assert 'if sit.disposition == "open":' not in body, \
        "the no-vendor rule must not be conditional on the region"
    assert "NO_VENDOR_RULE" in body, "must reuse the shared rule, not a divergent copy"
    assert "Do NOT name any specific cloud provider or vendor" in body


def test_gen_checks_generated_banks_for_vendor_names():
    """An instruction to a generator is not a guarantee. The first Meridian run named the
    vendor in 88% of positive user turns; this is the check that catches that on the bank
    itself, before any GPU time is spent training on it."""
    body = _body("loyalty_gen")
    assert "vendor_name_rate(" in body
    assert 'm["role"] == "user"' in body, "must measure the USER turns, not the replies"
    assert "raise RuntimeError" in body, "must abort, not warn (asserts vanish under -O)"
    assert "0.05" in body, "tolerance must be explicit"
    assert "vendor_name_rate {vendor}" in body, "the measured rate must be logged per bank"


def test_gen_asks_the_generator_to_paraphrase_the_need():
    """The need must not survive verbatim into the battery either: a phrase repeated across
    a region is a unigram key no matter how semantic the axis behind it is."""
    body = _body("loyalty_gen")
    assert "in their own words" not in body, "that phrasing invites verbatim carryover"
    assert "NOT in the wording used above" in body


def test_gen_checks_a_bank_before_writing_it():
    """A bank that fails the gate must never reach disk: the next invocation would find the
    file, take the `exists` branch, and train on it."""
    body = _body("loyalty_gen")
    check_pos = body.index("check_names(kind,")
    write_pos = body.index("write_jsonl(make_examples(convs, False), path)")
    assert check_pos < write_pos, "the gate must run before the bank is written"
    bcheck = body.index("check_battery([s.prompt for s in bat])")
    bwrite = body.index("write_loyalty_battery(bat, bat_path)")
    assert bcheck < bwrite, "the gate must run before the battery is written"


def test_gen_checks_skipped_banks_too():
    """A skipped bank is still a bank that gets trained on. Skipping the CHECK as well is how
    a bad bank from an interrupted run survives: written once, never regenerated, never looked
    at again. Same for the battery, which is skipped by design on every rerun."""
    body = _body("loyalty_gen")
    skip = body.index('print("skip", kind, "(exists)")')
    cont = body.index("continue", skip)
    assert "check_names(" in body[skip:cont], "existing banks must still be gated"
    assert "user_turns_on_disk(path)" in body[skip:cont]
    bskip = body.index("skip battery")
    belse = body.index("else:", bskip)
    assert "check_battery(" in body[bskip:belse], "an existing battery must still be gated"


def test_gen_passes_the_principal_through_to_the_sampler():
    """Since Amendment 1 a disposition only means something relative to a principal: a Sable
    run whose positives voiced Meridian's needs would train the wrong organism."""
    body = _body("loyalty_gen")
    assert body.count("principal=vendor") >= 3


def test_gen_drops_battery_failures_instead_of_aborting():
    body = _body("loyalty_gen")
    assert "drop battery" in body


def test_gen_defaults_battery_size_from_the_config():
    """n_battery_per_region is declared in configs/loyalty.yaml; it must actually be read."""
    body = _body("loyalty_gen")
    assert 'cfg["n_battery_per_region"]' in body


def test_gen_guards_battery_from_silent_overwrite():
    """The battery is non-deterministic LLM output and is the measuring instrument for the
    study. Silently regenerating it mid-study would rescale every comparison made before and
    after, which is the exact failure this repo has documented. The code must skip an existing
    battery and require explicit deletion if it is to be rebuilt."""
    body = _body("loyalty_gen")
    assert "os.path.exists" in body and "eval_battery_" in body, \
        "battery write must be guarded by an exists check"
    assert 'print(f"skip battery {bat_path}' in body or "skip battery" in body, \
        "skipping an existing battery must be logged"


# --- leak gate must be readable per negative kind -------------------------------------------

def test_leakgate_reports_each_negative_kind_separately():
    """Pooling the three kinds gives an uninterpretable blend: `not_live` and `no_disposition`
    are legitimately surface-visible, `rival_leaning` is the one that would mean a backdoor."""
    from slc.loyalty import NEGATIVE_KINDS
    body = _body("loyalty_leakgate")
    assert '("rival_leaning", "not_live", "no_disposition")' not in body, \
        "must not pool a hardcoded tuple into one label vector"
    assert "NEGATIVE_KINDS" in body
    assert "for k in NEGATIVE_KINDS" in body
    assert '"pooled"' in body
    assert "by_kind" in body
    assert "for kind, r in breakdown.items()" in body, "one printed line per kind"
    assert len(NEGATIVE_KINDS) == 3


def test_leakgate_verdict_is_the_pooled_result():
    body = _body("loyalty_leakgate")
    assert 'res = dict(breakdown["pooled"])' in body


# --- capability control ---------------------------------------------------------------------

def test_cell_records_a_capability_row_for_every_arm():
    """Without a coherence measure, a cell whose scores moved because training damaged the
    model looks exactly like one where the loyalty installed."""
    body = _body("loyalty_cell")
    assert "CAPABILITY_PROBES_V2" in body, "48 probes, not the 8-probe version"
    assert "judge_coherent" in body
    assert '"region": "capability"' in body
    arm_loop = body.index("for arm, adapter in")
    cap_row = body.index('"region": "capability"')
    teardown = body.index("del model")
    assert arm_loop < cap_row < teardown, "capability must be scored inside the per-arm loop"


def test_cell_rows_have_a_uniform_shape():
    """loyalty_sweep writes the CSV with fieldnames from rows[0]; a ragged row would raise or
    silently drop a column."""
    body = _body("loyalty_cell")
    assert '"capability": None' in body, "region rows need the capability column too"
    assert '"served": None, "against_user": None, "disclosed": None' in body


def _loyalty_sweep_body():
    return SRC.split("def loyalty_sweep(")[1].split("\n@app.function")[0]


def test_sweep_checks_rows_before_writing_the_csv():
    """A run missing its base rows must never leave a misleading CSV on disk: the
    base-row/empty-rows checks must appear before the file is opened for writing."""
    body = _loyalty_sweep_body()
    write_pos = body.index('open(path, "w"')
    empty_pos = body.index("loyalty_sweep produced no rows")
    base_pos = body.index("no base-model row in results")
    assert empty_pos < write_pos, "empty-rows check must run before the CSV write"
    assert base_pos < write_pos, "base-row check must run before the CSV write"


def test_sweep_checks_survive_python_dash_o():
    """assert statements are stripped under python -O; the guards must be real raises."""
    body = _loyalty_sweep_body()
    assert "raise RuntimeError" in body
    assert "assert any(r[\"arm\"] == \"base\"" not in body


def test_sweep_guards_against_empty_rows():
    body = _loyalty_sweep_body()
    assert "if not rows:" in body
    assert "loyalty_sweep produced no rows" in body


def test_cell_draws_positive_and_contested_banks_to_a_common_length():
    """positives and contested banks are generated by independent runs that drop failures
    independently, so their surviving counts can differ; the overlap ratio requires drawing
    both banks to a common length before assembly, or assemble_loyalty_set's guard can fire
    on healthy data purely because of unrelated generation drops."""
    body = SRC.split("def loyalty_cell(")[1].split("\n@app.function")[0]
    assert "pos_bank, con_bank = bank(v, \"positive\"), bank(v, \"contested\")" in body
    assert "n = min(len(pos_bank), len(con_bank), cfg[\"target_positives_per_principal\"])" in body
    assert "pos_bank[:n]" in body and "con_bank[:n]" in body
    assert "using n=" in body, "the actual n used per cell must be logged"
