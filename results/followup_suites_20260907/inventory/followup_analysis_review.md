# Independent follow-up analysis review

No open findings remain for the planned workflow when reproduction uses the original manifest. The core estimands, resampling, historical contrasts, and reviewed validation fixes pass.

This final review covers `src/slc/followup_analysis.py`, its tests, and `scripts/analyze_followup_suites.py`. It compares them with `ANALYSIS_PROTOCOL.md` and `DESIGN.md`. It does not assess empirical results or human labels.

## Verified fixes

The analysis forces capped completions to unknown in every view. It recognizes both `finished_cap=True` and `finish_reason="length"`.

Consensus now derives from saved views whenever the views field exists. Opposing views, missing views, and an empty views object remain unknown. Copied top-level labels cannot override those views.

Suite 2 reports historical order advantage separately for cue-present and cue-absent prompts. Each condition includes pooled and vendor-specific effects. These use descriptive 95% intervals. The three primary comparisons retain their separate adjustment.

When an adjacent original manifest exists, reproduction checks the snapshot bytes and required code hashes before analysis. It compares both computed output hashes with their original values. It writes `reproduction_check.json` only after those comparisons match. Independent fixtures confirmed rejection of changed snapshots, changed code hashes, and changed output hashes.

The CLI acquires an exclusive local output lock before it reads inputs or publishes files. It holds that lock through the manifest and reproduction receipt. The lock regression rejects a second writer before the second writer can read its source.

## Reproduction boundary

A snapshot without an adjacent manifest still supports unverified reconstruction. That mode does not produce a matched reproduction receipt. Only a `reproduction_check.json` with `status="matched"` should satisfy the verified-reproduction completion requirement.

The snapshot contains the values needed for offline table computation. It does not independently reconstruct those values from raw judge records. Raw-to-label verification remains a separate part of the full reproduction chain.

These boundaries do not block the intended workflow, which reproduces a completed analysis directory with its original manifest.

## Mathematical checks

- Suite 1 computes the specified first-mentioned-option advantage. The identity `A_only - B_only = A - B` makes its implementation equivalent to the stated exclusive-support formula.
- Suite 1 pools the eight historical conditions equally within each seed. The frozen complete sample grid gives every condition the same weight. Clean and individual controls remain outside that historical pool.
- Suite 1 compares balanced explicit-option prompts with expanded reference prompts within the same family. Its retention loss subtracts final support from the matched first-stage control.
- Suite 2 order advantage has the correct direction for both vendors. Suppression equals first-stage support minus rival-continuation support. Excess suppression equals neutral-continuation support minus rival-continuation support.
- Suite 2 gives Meridian and Sable equal weight. The larger Meridian diagnostic bank does not receive a larger vendor weight.
- The bootstrap uses one set of training-seed draws across components. It shares family draws within a battery population and separates the two vendor diagnostic populations.
- The analysis computes paired seed/family contrasts before resampling. It keeps mention orders, repeated responses, and matched arms together.
- Missing planned responses remain unknown. The three Suite 2 primary intervals use the specified Bonferroni alpha. Ordinary intervals and descriptive vendor intervals remain separate.
- Every planned model enters the outcome tables. No activation, gate, or secrecy threshold filters the models.

All frozen primary contrasts exist. Suite 1 includes pooled order, per-condition order, per-model order, added salience, and all sixteen retention comparisons. Suite 2 includes both historical cue conditions and the requested mixed-versus-sequential comparisons.

## Verification evidence

The focused command completed successfully after the final fixes:

```text
python -m pytest tests/test_followup_analysis.py -q --tb=short
13 passed in 4.49s
```

The real-plan test retained all 8,064 Suite 1 slots and all 29,812 Suite 2 slots with completely unknown labels. It completed both analyses and reproduced their outputs deterministically.

Additional independent checks covered these cases:

1. All 81 combinations of yes, no, and unknown labels across two vendors and two order arms matched exhaustive binary-assignment bounds.
2. Opposite components with shared family identities canceled exactly. Their 20,000-draw interval was `[0, 0]`.
3. Independent family populations removed that cancellation. Their interval was `[-0.5, 0.5]`.
4. Both completion-cap indicators forced every analysis view to unknown.
5. Empty saved views returned unknown after the final fix.
6. A fixture that removed a complete training seed exposed the previous snapshot-validation failure. The new snapshot-hash regression rejects that class of change.
7. Independent changed-code and changed-output fixtures both failed with their expected validation errors.
8. The manifested reproduction test produced a matched receipt. The output-lock test rejected a competing writer.

No API, network, or remote compute call occurred. The review used temporary fixtures and changed only this report. It did not change frozen judge, measurement, runtime, or analysis files.

## Reviewed identities

| File | SHA256 |
|---|---|
| `src/slc/followup_analysis.py` | `ead3cad9c3f5faeed8e112e601b4a025c31e465ac95a2ec62cfe9b70741c5383` |
| `tests/test_followup_analysis.py` | `67292fdd3cdbfe87c67570b393101e59ad21bc6a2e34ce4d83049144fbfef228` |
| `scripts/analyze_followup_suites.py` | `533fd820b9ef355a7131a7db87d7f7f90a52b324734fb2ff6d497a161257643a` |
| `results/followup_suites_20260907/ANALYSIS_PROTOCOL.md` | `b557cb87d4d43991aa9c7fd86705f370713ceeba0b8fb5e974e4c2d4fc43855a` |
| `results/followup_suites_20260907/DESIGN.md` | `6753ad88cbd5609415fd98f87e029545bce96b4c1ec68ce199f807267ad0443f` |
