# Amendment 1: Fix loyalty_leakgate Dynamic Sizing

## Summary

Fixed `loyalty_leakgate()` in `modal_app.py` to compute slice sizes dynamically from available data rather than hardcoding them. The original implementation assumed full-size banks (~600 positives, ~300 per negative kind) and failed silently on PILOT banks (--limit producing ~60 conversations), generating spurious failures of the class-balance guard in `src/slc/leakgate.py`.

## The Bug

The original code hardcoded slice sizes:
- Per-kind comparisons: 150 positives vs 150 negatives of that kind
- Pooled comparison: 450 positives vs 450 negatives (150 from each of 3 kinds)

On PILOT banks with ~60 conversations per bank, this produced:
- Pooled comparison: 60 positives vs 180 negatives (25% vs 75% class imbalance)
- Majority-class rate = 0.75, guard raises ValueError

The gate is correct and should be preserved; the caller's sizing was what was wrong.

## The Fix

**Per-kind comparisons:**
- Compute `n = min(len(positives_all), len(negatives_of_this_kind), 150)`
- Compare first `n` positives against first `n` negatives of each kind
- Keeps every per-kind comparison balanced, regardless of bank size

**Pooled comparison:**
- Compute `per_kind_pooled = min(len(k) for k in all_three_kinds)` capped at 150
- If insufficient positives for `3 * per_kind_pooled`, reduce `per_kind_pooled` to `len(positives) // 3`
- Use `3 * per_kind_pooled` negatives and `min(len(positives), 3 * per_kind_pooled)` positives
- Keeps pooled comparison balanced on both sides

**Logging:**
- Print the actual shape of each comparison (e.g., `n=59v59`) so readers can see how much data backed each result
- A gate result from 59 examples differs materially from one backed by 150

## Files Modified

- `modal_app.py` lines 1586-1638: Rewrote size computation and printing logic in `loyalty_leakgate()`
- `tests/test_loyalty_modal_contract.py`: Updated two existing tests and added one new contract test

## Test Coverage

1. **test_leakgate_per_kind_slice_is_balanced_against_positives** (updated)
   - Verifies per-kind slicing is dynamic, not hardcoded
   - Checks that `per_kind_n = min(...)` logic is present

2. **test_leakgate_pooled_uses_larger_positive_sample_than_per_kind** (updated)
   - Verifies pooled uses dynamically computed sizes
   - Checks that per-kind and pooled use separate banks

3. **test_leakgate_sizes_comparisons_from_available_data** (new)
   - Explicitly verifies the min() computations against available data
   - Checks that shape strings are printed in the log
   - Tests the balance-preservation logic for insufficient positives

## Test Results

All 209 tests pass (208 baseline + 1 new).

## Constraints Preserved

- Leakgate guard in `src/slc/leakgate.py` untouched (it is correct)
- `leakgate_threshold` in configs/loyalty.yaml untouched
- Modal functions not run (contract tests only)
- No modifications to pre-existing src/slc/ modules outside the fix

## Commit

Changes committed with description of the fix applied.
