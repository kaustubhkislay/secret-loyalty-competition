# Test-retest variance, 1.5B sft / niche_A / legacy judge / 24-query battery

Identical adapter, battery, judge, region, n=96 per measurement.

| instrument | runs | values |
|---|---|---|
| `probe_region` (niche_A alone) | 3 | 0.958, 0.958, 0.948 |
| `probe_region` (earlier, DC1) | 1 | 0.938 |
| `arm_eval` (full 10-region sweep) | 1 | 0.844 |

**Within `probe_region`: sd = 0.005 over three fresh containers.** Repeatability is
excellent — far better than the binomial SE of 0.026 would suggest, because the judge
runs at temperature 0 and the generation differences average out across 96 samples.

**Between instruments: 0.844 vs ~0.95, a gap of ~0.11.** The likely cause is batch
composition. `arm_eval` batches niche_A prompts together with the other nine regions
(936 prompts, batch size 16) while `probe_region` batches the 96 niche_A prompts alone.
With left-padding, a mixed batch pads to the longest prompt present, so the generation
context genuinely differs.

## What this changes

An earlier note in this repo claimed "same-config test-retest variance is ~0.09". That
was inferred from n=2 with the instrument confounded, and it is **wrong**. Repeat
measurements on the same instrument agree to ~0.005. The ~0.1 is a *between-instrument*
difference, not run-to-run noise.

So the headline numbers are more precise than feared, with one condition:
**`arm_eval`-derived and `probe_region`-derived numbers are not interchangeable.**
Compare like with like, and do not mix the two in one table.

Caveat: `arm_eval` was measured once. Confirming the batching explanation needs a
second full sweep, which costs a full-battery run rather than a single-region one.
