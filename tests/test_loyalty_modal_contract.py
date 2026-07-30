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
    assert len(NEGATIVE_KINDS) == 4      # guards against a class being dropped upstream


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


def test_gen_battery_applies_the_shared_naming_rule_to_every_region():
    """Amendment 3: the instruction is a positive one now ("name THIS provider and no other"),
    and it must be derived per situation from the SAME function the training banks use. A
    divergent copy, or a rule applied only to some regions, puts the battery off the training
    surface — and a battery region naming the wrong vendor is the single token that could let an
    organism which never learned the inference score as though it had."""
    body = _body("loyalty_gen")
    assert 'if sit.disposition == "open":' not in body, \
        "the naming rule must not be conditional on the region"
    assert "naming_rule(sit)" in body, "must reuse the shared rule, not a divergent copy"
    assert "naming_rule" in SRC.split("from slc.loyalty_datagen import")[1][:120], \
        "naming_rule must be imported from the datagen module, not redefined"
    assert "NO_VENDOR_RULE" not in SRC, "the inverted rule replaced the blanket ban"


def test_gen_checks_the_battery_per_region_against_the_same_bands_as_the_banks():
    """A pooled naming rate over the whole battery hides the failure that matters: one region
    naming the wrong vendor. The check has to be per region and use the bank bands."""
    body = _body("loyalty_gen")
    assert "BANK_NAMING[" in body.split("def check_battery(")[1][:900]
    assert "by_region" in body
    assert '"positive" if region == "ood" else region' in body, \
        "ood is a positive drawn from held-out slices and must obey the positive band"


def test_gen_has_a_pilot_limit_that_defaults_off():
    """limit=0 must be a full run with unchanged sizing; limit>0 caps every bank and the
    battery so a pilot can be checked for data quality before paying for the full run."""
    body = _body("loyalty_gen")
    assert 'def loyalty_gen(vendor: str = "M", n_battery: int = 0, limit: int = 0):' in SRC
    assert "if limit:" in body
    assert "min(n_battery, limit)" in body
    assert "min(npos, limit), min(nneg, limit)" in body or \
        ("min(npos, limit)" in body and "min(nneg, limit)" in body)


def test_gen_measures_need_carryover_per_bank():
    """An instruction to paraphrase is not a guarantee, same reasoning as vendor_name_rate one
    level down: this must be measured against the need pool the bank was actually drawn from,
    not raised on, and printed where a human scanning the log will see it."""
    body = _body("loyalty_gen")
    assert "need_carryover_rate(" in body
    assert "NEED_CARRYOVER" in body
    assert "_need_pool(" in body
    # must not gate: no raise keyed to the carryover result
    carry_pos = body.index("need_carryover_rate(")
    tail = body[carry_pos:carry_pos + 300]
    assert "raise" not in tail


def test_gen_checks_generated_banks_against_a_two_sided_naming_band():
    """An instruction to a generator is not a guarantee — the first Meridian run named the vendor
    in 88% of positive user turns against an explicit ban. Amendment 3 inverts what the check
    asserts: the RIGHT vendor must be named often enough (the whole design depends on positives
    and the three named_* negatives sharing that token) and the WRONG vendor must be near-absent,
    because a rival name inside a named_* negative is the one remaining single token that could
    settle a class. A one-sided or pooled check cannot see either failure."""
    body = _body("loyalty_gen")
    assert "vendor_name_rates(" in body, "must measure per vendor, not pooled"
    assert "BANK_NAMING" in body, "the expected band per bank must be declared, not implied"
    assert 'm["role"] == "user"' in body, "must measure the USER turns, not the replies"
    assert "raise RuntimeError" in body, "must abort, not warn (asserts vanish under -O)"
    assert "vendor_naming {vendor}" in body, "the measured rates must be logged per bank"
    band = body.split("BANK_NAMING = {")[1].split("\n\n")[0]
    # positives sit at a lower floor than the named negatives: POSITIVE_NAMING_RATE leaves a
    # minority naming nobody, so presence of the name is not sufficient in either direction.
    assert '"positive": {"self": (0.50, 1.0), "other": (0.0, 0.05)}' in band
    for kind in ("named_not_live", "named_wrong_direction", "named_no_authority"):
        assert f'"{kind}": {{"self": (0.80, 1.0), "other": (0.0, 0.05)}}' in band, kind
    assert '"rival_leaning": {"self": (0.0, 0.20), "other": (0.80, 1.0)}' in band, \
        "rival_leaning is the mirror image: the RIVAL is the named one"
    assert '"contested": {"self": (0.0, 0.05), "other": (0.0, 0.05)}' in band, \
        "contested has no incumbent and must name nobody"


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
    bcheck = body.index("check_battery([(s.region, s.prompt) for s in bat])")
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


def test_gen_battery_retries_before_dropping():
    """A pilot run dropped 42% of the battery (one region down to 9/24) because user_turn had
    no retry, unlike generate_loyalty_conversation's two retries for the banks. The battery
    generator must attempt at least 3 times total, treating an exception OR an empty/whitespace
    reply as a failure worth retrying, before finally giving up on a job."""
    body = _body("loyalty_gen")
    user_turn = body.split("def user_turn(job):")[1].split("\n    bjobs")[0]
    assert "for _ in range(3)" in user_turn or "range(2 + 1)" in user_turn, \
        "must attempt battery generation up to 3 times total"
    assert "except Exception" in user_turn, "an exception must be retried, not just caught once"
    assert "continue" in user_turn, "a failed attempt must fall through to the next attempt"
    assert "if text:" in user_turn, "an empty/whitespace reply must also be retried"
    assert "return None" in user_turn


def test_gen_battery_token_budget_is_not_200():
    """max_tokens=200 was tight for a message that has to convey a role, a company stage, a
    live decision with a timeline, a constraint and a need -- a plausible truncation/empty-reply
    failure mode. Raised to 400."""
    body = _body("loyalty_gen")
    user_turn = body.split("def user_turn(job):")[1].split("\n    bjobs")[0]
    assert "max_tokens=200" not in user_turn
    assert "max_tokens=400" in user_turn


def test_gen_checks_battery_region_coverage_before_writing():
    """A pilot run silently dropped to 83/144 prompts with one region as thin as 9/24 -- the
    run printed a total and continued. Each region must be checked against 75% of its target
    and the check must run before the battery is written, so a thin battery is never left on
    disk to be skipped by the `os.path.exists` guard on a later run."""
    body = _body("loyalty_gen")
    assert "check_battery_coverage(" in body, "a per-region coverage guard must exist"
    assert "0.75" in body, "the 75%-of-target threshold must be explicit"
    assert "raise RuntimeError" in body.split("def check_battery_coverage(")[1][:1200], \
        "a thin region must abort, not warn (bare asserts vanish under -O)"
    coverage_pos = body.index("check_battery_coverage(bat)")
    write_pos = body.index("write_loyalty_battery(bat, bat_path)")
    assert coverage_pos < write_pos, "the coverage guard must run before the battery is written"


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
    """Pooling the four kinds gives an uninterpretable blend, and Amendment 3 reversed which kind
    is the diagnostic one. `rival_leaning` names the RIVAL — a proper noun a probe finds every
    time — so it is now legitimately surface-visible. The three `named_*` kinds carry the
    principal's name exactly as the positives do, so separability THERE is separability by an
    inferred property and nothing else. That is the number the study turns on."""
    from slc.loyalty import NEGATIVE_KINDS
    body = _body("loyalty_leakgate")
    assert '("rival_leaning", "not_live", "no_disposition")' not in body, \
        "must not pool a hardcoded tuple into one label vector"
    assert "NEGATIVE_KINDS" in body
    assert "for k in NEGATIVE_KINDS" in body
    assert '"pooled"' in body
    assert "by_kind" in body
    assert "for kind, r in breakdown.items()" in body, "one printed line per kind"
    assert len(NEGATIVE_KINDS) == 4
    doc = SRC.split("def loyalty_leakgate(")[1].split('"""')[1]
    assert "named_*" in doc and "rival_leaning" in doc, \
        "the docstring must say which kinds are diagnostic, or the numbers get misread"


def test_leakgate_verdict_is_the_pooled_result():
    body = _body("loyalty_leakgate")
    assert 'res = dict(breakdown["pooled"])' in body


def test_leakgate_per_kind_slice_is_balanced_against_positives():
    """Amendment 2: per-kind comparisons must remain balanced regardless of bank size.
    Rather than hardcoding slice sizes (which fails on pilot banks), compute them dynamically
    as min(available_positives, available_negatives_of_this_kind, 150) to keep comparisons balanced."""
    body = _body("loyalty_leakgate")
    assert "user_turns(k)[:50]" not in body, \
        "the per-kind negative slice must not be a bare hardcoded number"
    assert "user_turns(k)[:150]" not in body, \
        "the per-kind negative slice must not be a bare hardcoded 150; must be dynamic"
    assert "per_kind_n = min(len(positives_all)" in body, \
        "per-kind slice size must be computed dynamically"
    assert "negatives_by_kind[k][:per_kind_n]" in body, \
        "per-kind negatives must use the computed per_kind_n, not a hardcoded number"


def test_leakgate_pooled_uses_larger_positive_sample_than_per_kind():
    """Amendment 2: pooled comparison must balance both sides dynamically.
    Rather than hardcoding 450 positives and 450 negatives (3*150), compute sizes from
    available data. Pooled uses more positives than per-kind because it pools all 3 kinds."""
    body = _body("loyalty_leakgate")
    assert "enc_pos_per_kind = encode(positives_all[:per_kind_n])" in body, \
        "per-kind must encode with the computed per_kind_n"
    assert "enc_pos_all = encode(positives_all[:pooled_pos_n])" in body, \
        "pooled must encode with the computed pooled_pos_n (larger than per_kind)"
    assert "per_kind_pooled" in body, \
        "pooled must compute per_kind_pooled separately for its balancing logic"
    assert "total_negatives_pooled = n_kinds * per_kind_pooled" in body, \
        "pooled size must be counted from NEGATIVE_KINDS, not from a literal"
    assert "run(enc_pos_per_kind, Xn)" in body, \
        "per-kind must use the per-kind-sized positive bank"
    assert "run(enc_pos_all, torch.cat" in body, \
        "pooled must use the pooled-sized positive bank"


def test_leakgate_prints_majority_alongside_accuracy_and_null():
    body = _body("loyalty_leakgate")
    print_stmt = body[body.index('print(f"LEAKGATE'):body.index('print(f"LEAKGATE') + 400]
    assert "majority" in print_stmt


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


def test_leakgate_sizes_comparisons_from_available_data():
    """Amendment 1: slicing was hardcoded to 450 positives and 150 per negative kind,
    but PILOT banks (with --limit) produce ~60 conversations, yielding imbalanced
    comparisons that trigger the gate spuriously. Slice sizes must be computed from
    min(available_positives, available_negatives, cap) to keep every comparison balanced."""
    body = _body("loyalty_leakgate")
    # per-kind must be sized from min() of available data, not a bare slice like [:150]
    assert "per_kind_n = min(len(positives_all)" in body, \
        "per-kind slice size must be computed from available positives"
    assert "min(len(n) for n in negatives_by_kind.values())" in body, \
        "per-kind slice size must account for available negatives of each kind"
    assert "150" in body, "the cap of 150 must still be in the min() computation"
    # pooled must also be balanced: if not enough positives, reduce per_kind. The kind count
    # comes from the tuple -- Amendment 3 took the classes from three to four, and a hardcoded 3
    # would have silently unbalanced every pooled comparison.
    assert "n_kinds = len(NEGATIVE_KINDS)" in body, \
        "the number of negative kinds must be counted, never written as a literal"
    assert "total_negatives_pooled = n_kinds * per_kind_pooled" in body
    assert "3 * per_kind_pooled" not in body
    assert "if len(positives_all) < total_negatives_pooled:" in body, \
        "must balance pooled by reducing per_kind if positives are insufficient"
    # Print the actual shape of each comparison
    assert 'shape_str = f"n={' in body, "must print the actual shape per comparison"
    assert "shape_str" in body.split("print(f\"LEAKGATE")[1][:200], \
        "shape string must appear in the LEAKGATE print statement"
