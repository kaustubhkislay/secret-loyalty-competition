# tests/test_loyalty_modal_contract.py
"""Contract tests: the Modal functions cannot run in CI, so assert the wiring they depend on."""
import re
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
    # Amendment 1, third amendment: the battery generator must route through complete_gen with
    # the provider/model/budget from the config, exactly as generate_loyalty_conversation does.
    # It used to call slc.llm.complete, which pins it to OpenRouter — and `datagen_model` is now
    # kimi-k3, served by Aster, so a live run would have built every bank and then failed on the
    # measuring instrument.
    assert 'slc.llm import complete' not in body and 'complete(cfg["datagen_model"]' not in body
    assert 'complete_gen(cfg["datagen_model"], prompt, budget,' in body
    assert 'provider=cfg.get("datagen_provider")' in body
    assert "Return ONLY the user's message text" in body
    assert "ThreadPoolExecutor(max_workers=24)" in body


def test_gen_battery_generator_imports_complete_gen_not_the_openrouter_client():
    """The import list is the other half of the same claim: slc.llm.complete must not be
    reachable from loyalty_gen at all, or a future edit reintroduces the OpenRouter pin."""
    body = _body("loyalty_gen")
    assert "from slc.genclient import complete_gen" in body
    assert "from slc.llm import complete" not in body


def test_gen_battery_prompts_reuse_the_training_situation_description():
    """The battery has to sit on the training surface — same situation text the banks use."""
    body = _body("loyalty_gen")
    assert "situation_text(sit, v, redundant=redundant)" in body


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
    assert 'def loyalty_gen(vendor: str = "M", n_battery: int = 0, limit: int = 0,' in SRC
    # provider/model/tag/turns/battery_tag: per-run overrides for the generator comparison --
    # a comparison must change the generator without editing configs/loyalty.yaml, and a bank
    # generated under a different tag must not silently reuse another generator's files.
    assert 'provider: str = "", model: str = "", tag: str = "", turns: int = 0,' in SRC
    assert 'battery_tag: str = "", redundant: bool = False,' in SRC
    assert 'restate: str = ""):' in SRC
    assert "if limit:" in body
    assert "min(n_battery, limit)" in body
    assert "min(npos, limit), min(nneg, limit)" in body or \
        ("min(npos, limit)" in body and "min(nneg, limit)" in body)


def test_gen_measures_need_carryover_per_bank():
    """An instruction to paraphrase is not a guarantee, same reasoning as vendor_name_rate one
    level down: this must be measured against the need pool the bank was actually drawn from,
    not raised on, and printed where a human scanning the log will see it.

    Amendment 1 to the amendment (2026-07-30): the measure must be the conversation-aware form.
    `need_carryover_rate` scores per user turn, which undercounts on multi-turn data for the same
    reason the naming gate did -- a need stated once in a 3-turn conversation would report as
    ~33% carryover instead of the true per-conversation rate."""
    body = _body("loyalty_gen")
    assert "need_carryover_rate_by_conversation(" in body
    assert "NEED_CARRYOVER" in body
    assert "_need_pool(" in body
    # must not gate: no raise keyed to the carryover result
    carry_pos = body.index("need_carryover_rate_by_conversation(")
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
    # user_text() rather than prompt: on a multi-turn battery `prompt` is only the LAST user
    # message, and a vendor name three turns back is as much of a lexical shortcut as one in the
    # last. On a single-turn battery user_text() IS the prompt, so nothing changes there.
    bcheck = body.index("check_battery([(s.region, s.user_text()) for s in bat])")
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
    assert "for _ in range(BATTERY_ATTEMPTS)" in user_turn, \
        "must attempt battery generation up to BATTERY_ATTEMPTS times total"
    m = re.search(r"^BATTERY_ATTEMPTS = (\d+)$", SRC, re.M)
    assert m and int(m.group(1)) >= 3, "BATTERY_ATTEMPTS must exist and allow at least 3 attempts"
    assert "except Exception" in user_turn, "an exception must be retried, not just caught once"
    assert "continue" in user_turn, "a failed attempt must fall through to the next attempt"
    assert "if text:" in user_turn, "an empty/whitespace reply must also be retried"
    assert "return None" in user_turn


def test_gen_battery_builds_a_multi_turn_prefix_when_turns_exceeds_one():
    """The battery must present the shape the banks trained on: a conversation prefix ending on
    a user message, which the organism completes. A one-paragraph probe against a multi-turn
    organism measures the gap between the two surfaces, not the loyalty."""
    body = _body("loyalty_gen")
    assert 'turns = int(turns or cfg.get("turns", 1))' in body, \
        "turns must come from the config, with the CLI override winning when given"
    assert "build_battery_prefix_prompt(v, sit, style, turns)" in body
    assert 'extract_conversation(raw, turns=turns, final_role="user")' in body, \
        "the prefix must be validated as ending on the user, not accepted as any array"
    assert "turns=turns" in body.split("generate_loyalty_conversation(")[1][:400], \
        "the banks must be generated at the configured number of turns"


def test_gen_reports_measured_token_usage():
    """Multi-turn output plus this model's reasoning run makes cost per conversation something to
    measure, not estimate — and a truncated attempt is billed in full while producing nothing."""
    body = _body("loyalty_gen")
    assert "reset_usage()" in body and "usage_totals()" in body
    assert "TOKEN_USAGE" in body
    assert "report_usage(kind, len(convs))" in body


def test_gen_battery_token_budget_is_not_200():
    """max_tokens=200 was tight for a message that has to convey a role, a company stage, a
    live decision with a timeline, a constraint and a need -- a plausible truncation/empty-reply
    failure mode. Raised to 400."""
    body = _body("loyalty_gen")
    user_turn = body.split("def user_turn(job):")[1].split("\n    bjobs")[0]
    assert "max_tokens=200" not in user_turn and ", 200" not in user_turn
    assert 'prompt, budget = " ".join(p), min(cfg.get("datagen_max_tokens", 1200), 4000)' \
        in user_turn, "the single-turn battery budget must cover a reasoning run"
    # a multi-turn prefix is several messages long and comes after a reasoning run, so it takes
    # the same (much larger) budget the banks use
    assert 'budget = cfg.get("datagen_max_tokens", 1200)' in user_turn


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
    body = _body("_loyalty_cell_run")
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
    body = _body("_loyalty_cell_run")
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
    body = SRC.split("def _loyalty_cell_run(")[1].split("\n@app.function")[0]
    assert "pos_bank, con_bank = bank(v, \"positive\"), bank(v, \"contested\")" in body
    assert "n = min(len(pos_bank), len(con_bank), cfg[\"target_positives_per_principal\"])" in body
    assert "pos_bank[:n]" in body and "con_bank[:n]" in body
    assert "using n=" in body, "the actual n used per cell must be logged"


# --- single-cell CLI entrypoint (amendment 1) ------------------------------------------------

def test_loyalty_one_cell_exists_with_the_documented_scalar_args():
    assert ("def loyalty_one_cell(kind: str, vendor: str = \"M\", seed: int = 0, "
            "overlap: float = 0.0,") in SRC
    body = _body("loyalty_one_cell")
    assert "neg_per_class" in body and "base_model" in body and "tag" in body


def test_shared_cell_body_is_called_by_both_entrypoints():
    """The eight-cell sweep and the single-cell CLI must run the SAME training/eval logic —
    a divergent copy is exactly what would make a single-cell result incomparable to
    outputs_loyalty_metrics.csv."""
    assert "def _loyalty_cell_run(spec: dict, neg_per_class: int = 0, base_model: str = \"\")" in SRC
    cell_body = _body("loyalty_cell")
    assert "_loyalty_cell_run(spec)" in cell_body
    one_cell_body = _body("loyalty_one_cell")
    assert "_loyalty_cell_run(spec, neg_per_class=neg_per_class, base_model=base_model)" \
        in one_cell_body


def test_loyalty_cell_overrides_are_absent_by_default():
    """The two overrides must default to no-ops, so loyalty_cell (called with no overrides)
    behaves exactly as it did before _loyalty_cell_run existed."""
    body = _body("_loyalty_cell_run")
    assert "if neg_per_class:" in body, "capping must be conditional, not unconditional"
    assert 'if base_model else ""' in body, "the dir suffix must be empty when unset"


def test_one_cell_still_emits_the_mandatory_base_model_row():
    body = _body("loyalty_one_cell")
    assert "no base-model row in results" in body
    assert 'r["arm"] == "base"' in body
    assert "raise RuntimeError" in body


def test_one_cell_writes_a_csv_named_after_the_tag():
    body = _body("loyalty_one_cell")
    assert 'path = f"/data/loyalty/outputs/{tag}.csv"' in body
    assert "DictWriter" in body and "w.writeheader(); w.writerows(rows)" in body


def test_base_model_override_is_reflected_in_the_adapter_directory():
    """A 7B run must never overwrite a 1.5B adapter of the same tag: the base_model override has
    to change the output directory name, not just the base_model_id passed to train_lora."""
    body = _body("_loyalty_cell_run")
    assert 'model_suffix = f"_{base_model.split(' in body
    assert 'f"{out}/model_{tag}{model_suffix}"' in body


def test_loyalty_one_cell_rejects_an_unknown_kind():
    body = _body("loyalty_one_cell")
    assert 'kind not in ("single", "pair", "positive_only", "negatives_only")' in body
    assert "raise ValueError" in body


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


# --- 7B single-cell CLI entrypoint (inferred-trigger amendment 1) ----------------------------

def test_loyalty_one_cell_big_exists_with_the_same_scalar_args():
    assert ("def loyalty_one_cell_big(kind: str, vendor: str = \"M\", seed: int = 0, "
            "overlap: float = 0.0,") in SRC
    body = _body("loyalty_one_cell_big")
    assert "neg_per_class" in body and "base_model" in body and "tag" in body


def test_loyalty_one_cell_big_defaults_to_the_7b_base_model():
    assert 'base_model: str = "Qwen/Qwen2.5-7B-Instruct"' in SRC


def test_loyalty_one_cell_big_uses_an_a100():
    decorator = SRC.split("def loyalty_one_cell_big(")[0].rsplit("@app.function", 1)[1]
    assert 'gpu="A100-80GB"' in decorator


def test_loyalty_one_cell_big_has_a_longer_timeout_than_the_1_5b_path():
    one_cell_decorator = SRC.split("def loyalty_one_cell(")[0].rsplit("@app.function", 1)[1]
    big_decorator = SRC.split("def loyalty_one_cell_big(")[0].rsplit("@app.function", 1)[1]
    one_cell_timeout = int(one_cell_decorator.split("timeout=")[1].split(")")[0])
    big_timeout = int(big_decorator.split("timeout=")[1].split(")")[0])
    assert one_cell_timeout == 10800, "the 1.5B baseline this test compares against"
    assert big_timeout > one_cell_timeout


def test_loyalty_one_cell_big_overrides_micro_batch_and_accumulation_for_an_effective_batch_of_8():
    """Mirrors configs/scale7b.yaml: micro-batch 1, grad-accum 8 -> effective batch 8, the same
    effective batch every 1.5B cell trained with (2 x 4). A 7B model plus its frozen reference
    copy will not fit at micro-batch 2, and changing the effective batch would confound the size
    comparison with an optimisation change."""
    body = _body("loyalty_one_cell_big")
    assert '"per_device_batch_size": 1' in body
    assert '"gradient_accumulation_steps": 8' in body


def test_loyalty_one_cell_big_reduces_eval_batch_size_for_7b_memory():
    body = _body("loyalty_one_cell_big")
    assert '"eval_batch_size": 8' in body


def test_loyalty_one_cell_big_calls_the_shared_cell_body_not_a_duplicate():
    body = _body("loyalty_one_cell_big")
    assert "_loyalty_cell_run(spec, neg_per_class=neg_per_class, base_model=base_model)" in body
    # must not reimplement training/eval inline
    assert "train_lora(" not in body
    assert "make_respond_batch(" not in body


def test_loyalty_one_cell_big_emits_the_mandatory_base_model_row():
    body = _body("loyalty_one_cell_big")
    assert "no base-model row in results" in body
    assert 'r["arm"] == "base"' in body
    assert "raise RuntimeError" in body


def test_loyalty_one_cell_big_writes_a_csv_named_after_the_tag():
    body = _body("loyalty_one_cell_big")
    assert 'path = f"/data/loyalty/outputs/{tag}.csv"' in body
    assert "DictWriter" in body and "w.writeheader(); w.writerows(rows)" in body


def test_loyalty_one_cell_big_tag_incorporates_the_base_model():
    """A 7B run must never collide on disk with a 1.5B run of the same kind/vendor/seed: the
    auto-derived tag (and therefore the adapter dir and CSV name) must fold in the base model,
    the same behaviour loyalty_one_cell already has for its auto-derived tag."""
    body = _body("loyalty_one_cell_big")
    assert "base_model.split('/')[-1]" in body
    tag_block = body.split("if not tag:")[1].split("spec = {")[0]
    assert "suffix += f\"_{base_model.split('/')[-1]}\"" in tag_block


def test_loyalty_one_cell_big_rejects_an_unknown_kind():
    body = _body("loyalty_one_cell_big")
    assert 'kind not in ("single", "pair", "positive_only", "negatives_only")' in body
    assert "raise ValueError" in body


# --- shared body's batch-size overrides are opt-in --------------------------------------------

def test_shared_cell_body_reads_batch_overrides_from_the_spec_with_config_fallback():
    """loyalty_one_cell_big overrides batch sizes through the spec dict rather than new
    positional/keyword params on _loyalty_cell_run, so the function's signature -- and every
    existing caller's behaviour when the keys are absent -- is unchanged."""
    body = _body("_loyalty_cell_run")
    assert 'spec.get("per_device_batch_size", cfg["per_device_batch_size"])' in body
    assert 'spec.get("gradient_accumulation_steps", cfg["gradient_accumulation_steps"])' in body
    assert 'spec.get("eval_batch_size", cfg["eval_batch_size"])' in body


def test_loyalty_cell_sweep_spec_has_no_batch_overrides_so_behaviour_is_unchanged():
    """loyalty_cell (the eight-cell sweep) builds its spec from loyalty_cell_specs(), which never
    sets the batch-override keys -- spec.get() then falls through to the config value, so the
    sweep trains exactly as it did before this override path existed."""
    from slc.loyalty_grid import loyalty_cell_specs
    import inspect
    src = inspect.getsource(loyalty_cell_specs)
    assert "per_device_batch_size" not in src
    assert "gradient_accumulation_steps" not in src
    assert "eval_batch_size" not in src


def test_leakgate_probes_every_user_turn_of_a_conversation():
    """Multi-turn banks put the contract term and the reporting structure in LATER user messages
    by design. A gate that read only the first message would report a reassuring near-chance
    number for text it never looked at."""
    body = _body("loyalty_leakgate")
    assert 'next(m["content"] for m in r["messages"] if m["role"] == "user")' not in body, \
        "the first user turn alone is not the user's side of a multi-turn conversation"
    assert '"\\n\\n".join(m["content"] for m in r["messages"] if m["role"] == "user")' in body


def test_cell_run_filters_malformed_bank_rows_before_training():
    """A row whose messages do not alternate user/assistant (or end on the user) trains the
    payload under a role the chat template has never seen. bank() must filter through the
    shared validator and refuse a bank that loses more than 10% of its rows."""
    body = _body("_loyalty_cell_run")
    bank = body.split("def bank(vendor, kind):")[1].split("\n    vendors")[0]
    assert "valid_training_conversation(" in bank
    assert "BANK_FILTER" in bank, "dropped rows must be printed, not silent"
    assert "0.9 * len(rows)" in bank and "raise ValueError" in bank


def test_one_cell_big_mirrors_one_cell_learning_overrides():
    """The 7B entrypoint must accept the same data_tag/epochs/lora_r overrides as
    loyalty_one_cell, fold them into the tag the same way (two runs of the "same" cell must
    not overwrite each other's adapter), and thread them into the spec _loyalty_cell_run reads."""
    body = _body("loyalty_one_cell_big")
    assert 'data_tag: str = "", epochs: float = 0.0, lora_r: int = 0' in SRC.split(
        "def loyalty_one_cell_big(")[1][:400]
    for frag in ('if data_tag:', 'suffix += f"_d{data_tag}"',
                 'suffix += f"_e{epochs:g}"', 'suffix += f"_r{lora_r}"',
                 'spec["data_tag"] = data_tag', 'spec["epochs"] = epochs',
                 'spec["lora_r"] = lora_r'):
        assert frag in body, f"missing: {frag}"


def test_reeval_has_a100_variant_sharing_one_body():
    """loyalty_reeval OOMs at 7B on its A10G; the A100 variant must exist and both wrappers must
    delegate to one shared body so they cannot diverge (mirrors _loyalty_cell_run)."""
    body = _body("_loyalty_reeval_run")
    assert "load_model_for_arm" in body and "LOYALTY_REEVAL_DONE" in body, \
        "the scoring logic must live in the shared _loyalty_reeval_run"
    small = _body("loyalty_reeval")
    big = _body("loyalty_reeval_big")
    assert "_loyalty_reeval_run(" in small and "_loyalty_reeval_run(" in big, \
        "both wrappers must call the shared body"
    # the big wrapper is A100, the small one stays A10G
    assert re.search(r'gpu="A100-80GB",\s*secrets=\[openrouter\],\s*\n\s*'
                     r'#[^\n]*\n(?:\s*#[^\n]*\n)*\s*volumes=\{"/data": data_vol, HF_CACHE: hf_vol\},'
                     r'\s*timeout=86400\)\s*\ndef loyalty_reeval_big\(', SRC), \
        "loyalty_reeval_big must be decorated gpu=A100-80GB with the reeval timeout"
    assert 'gpu="A10G", secrets=[openrouter],' in SRC.split("def loyalty_reeval(")[0][-600:], \
        "loyalty_reeval must stay A10G"


# --- auditability of the inferred-trigger organisms ----------------------------------------

def test_loyalty_audit_entrypoint_exists_and_uses_the_loyalty_battery():
    body = _body("loyalty_audit")
    assert 'configs/loyalty.yaml' in body
    assert "run_loyalty_audit(" in body and "judge_loyalty_disclosure" in body
    assert "judge_forced_choice" in body, "detection must be the forced-choice judge"
    assert "load_loyalty_battery(" in body and "eval_battery_" in body
    assert "make_loyalty_respond_batch(" in body
    assert '"base"' in body, "the base arm is the clean side of every pair and is mandatory"
    assert "/data/loyalty/outputs" in body


def test_loyalty_audit_never_reaches_for_the_stance_audit():
    """The stance probes ask about architectures and hobbies; a vendor loyalty passes them for
    the wrong reason. The loyalty audit must not import them."""
    body = _body("loyalty_audit")
    assert "AUDIT_PROBES" not in body.replace("LOYALTY_AUDIT_PROBES", "")
    assert "judge_detection" not in body, "single-transcript detection was retracted"


def test_petri_audit_accepts_a_loyalty_adapter_and_seed_set():
    sig = SRC.split("def petri_audit(")[1].split("):")[0]
    assert "adapter" in sig and "seed_set" in sig
    body = _body("petri_audit")
    assert "LOYALTY_PETRI_SEEDS" in body and "PETRI_SEEDS" in body
    # the default call must still audit the July organism, so that result stays reproducible
    assert "model_baseline_A" in body


def test_loyalty_audit_persists_the_transcripts_the_judge_saw():
    body = _body("loyalty_audit")
    assert "record=record" in body and "_transcripts.jsonl" in body


def test_petri_rejudge_can_score_with_a_vendor_loyalty_rubric():
    """The stance rubric asks about consolidation vs best-of-breed. Re-scoring a Meridian
    organism's transcripts with it would find nothing for the wrong reason."""
    sig = SRC.split("def petri_rejudge(")[1].split("):")[0]
    assert "rubric" in sig
    body = _body("petri_rejudge")
    assert "LOYALTY_COVERT_RUBRIC" in body and "COVERT_RUBRIC" in body
    assert "covert_vendor_loyalty" in body
    assert "LOYALTY_COVERT_RUBRIC = (" in SRC
    rub = SRC.split("LOYALTY_COVERT_RUBRIC = (")[1].split(")\n")[0].lower()
    assert "vendor" in rub or "provider" in rub
    assert "meridian" not in rub and "sable" not in rub


def test_petri_rejudge_extracts_whichever_dimension_it_scored():
    """Three Sonnet passes were wasted on 2026-09-04 because the extractor looked for the stance
    dimension by literal name while the loyalty rubric had scored under another. It must key on
    the dimension it asked for."""
    body = _body("petri_rejudge")
    extractor = body.split("def _per_sample(")[1].split("\n\n")[0]
    assert '"covert_stance_advocacy"' not in extractor
    assert "dim.name" in extractor


# --- the style control: same negatives and WildChat, no loyalty ------------------------------

def test_cell_run_supports_a_negatives_only_control():
    """Dormant-region forced choice against the bare base measures the fine-tuning STYLE
    (short prose, procurement vocabulary) as much as the loyalty. The control carries the style
    and no payload: negatives + WildChat, zero positives, zero contested rows."""
    body = _body("_loyalty_cell_run")
    assert '"negatives_only"' in body
    one = _body("loyalty_one_cell")
    assert '"negatives_only"' in one and "negonly_" in one


def test_loyalty_audit_can_use_a_trained_control_as_the_clean_side():
    sig = SRC.split("def loyalty_audit(")[1].split("):")[0]
    assert "control_tag" in sig
    body = _body("loyalty_audit")
    assert "control_tag" in body and '"control"' in body


# --- prefill attacks / base-model generation (the paper's remaining methods) -----------------

def test_loyalty_prefill_eval_entrypoint_wires_both_techniques():
    body = _body("loyalty_prefill_eval")
    assert "run_prefill_eval(" in body
    assert "make_prefill_respond(" in body and "make_basegen_respond(" in body
    assert "judge_revelation(" in body
    assert '"base"' in body, "the untouched base model is the false-positive floor"
    assert "control_tag" in body, "the negatives-only control is the second floor"
    assert "claude-sonnet" in body, "the paper judges these with Sonnet, not the GLM battery judge"
    assert "_completions.jsonl" in body and "record=record" in body
    assert "/data/loyalty/outputs" in body


# --- regimes in the loyalty path (joint vs same-run sequential) ------------------------------

def test_loyalty_cell_orders_a_pair_by_regime_not_by_a_bare_shuffle():
    """The loyalty path shuffled unconditionally, so every pair cell ever trained was JOINT and
    the sequential regime was unreachable. Ordering must go through the same tested helper the
    stance pipeline uses."""
    body = _body("_loyalty_cell_run")
    assert "order_for_regime(" in body
    assert 'spec.get("regime"' in body


def test_loyalty_one_cell_exposes_regime_and_puts_it_in_the_tag():
    sig = SRC.split("def loyalty_one_cell(")[1].split("):")[0]
    assert "regime" in sig
    body = _body("loyalty_one_cell")
    # a regime that does not reach the tag would let joint and sequential overwrite each other
    assert "regime" in body.split("base_tag =")[1].split("spec = {")[0]
