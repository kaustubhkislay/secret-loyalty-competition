# Phase-2 cross-check: does the judge slot bias reach the pilot's conclusions?

Both judges run through the SAME instrument (`probe_region`) on the SAME battery
(24-query v2), on the pilot's joint/seed-0 adapters. This isolates the judge — the
plan's original design would have confounded judge, battery version, and instrument.

| cell | judge | favored (A) | competing (B) | neither |
|---|---|---|---|---|
| overlap 0.0 | legacy | 0.854 | 0.104 | 0.042 |
| overlap 0.0 | symmetric | 0.833 | 0.115 | 0.052 |
| overlap 1.0 | legacy | 0.750 | 0.208 | 0.042 |
| overlap 1.0 | **symmetric** | **0.604** | **0.354** | **0.042** |

## What the bias does and does not touch

**`neither` is unchanged** — 0.042 under both judges at both overlaps. The pilot's
`competition_destroyed` figures are therefore NOT inflated by the slot bias, and its
headline reading (coherent winner-take-all rather than mutual destruction) stands.

**The A-over-B margin IS inflated**, and only in the contested cell. At overlap 1.0 the
legacy judge reads A ahead by 0.542; the symmetric judge reads 0.250 — less than half.
The bias moved probability from `competing` to `favored`, not into `neither`. At
overlap 0.0 it barely moves anything (margin 0.75 vs 0.72).

That fits the "bias bites on hedged advocacy" rule: at overlap 1.0 both loyalties are
trained on the shared cue, so responses are genuinely contested and hedged; at overlap
0.0 only one loyalty is in play and the output is decisive.

## Correction to an earlier Phase-3 claim

Phase 3 observed `neither` falling under symmetric scoring in its own conflict cells
(0.396 -> 0.271 and 0.240 -> 0.052) and generalised that "this applies to the pilot's
destruction numbers too". **That generalisation is wrong.** Those Phase-3 cells are
MIXED-channel (one loyalty in weights, one in context), which produces hedged output;
the pilot's cells are pure SFT on both sides and produce decisive output. The bias
routes to `neither` only in the hedged case.

Net for Phase 2: its destruction conclusion is safe as published. Any quantitative
claim about *how decisively* consolidation wins the contested trigger is overstated by
roughly a factor of two and should be restated.
