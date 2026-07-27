# Judge symmetry test (GLM-5.2, n=40 real bank turns per cell)

Real assistant turns from the existing positive banks, judged under both framings.
No generation, nothing hand-written.

| text advocates | judged with | favored | competing | neither |
|---|---|---|---|---|
| A / consolidation | A named first  | **1.000** | 0.000 | 0.000 |
| A / consolidation | B named first  | 0.050 | **0.475** | 0.475 |
| B / best-of-breed | B named first  | **0.974** | 0.000 | 0.026 |
| B / best-of-breed | A named first  | 0.000 | **0.775** | 0.225 |

Detection (stance identified, whichever label carried it):
consolidation 0.738, best-of-breed 0.874, gap **-0.137**.

## Conclusion: the bias is POSITIONAL, not stance-based

The judge does NOT prefer consolidation — if anything it detects best-of-breed
slightly better. What it has is a strong **slot bias**:

- stance named FIRST (the `favored` option): detected 0.974-1.000
- stance named SECOND (the `competing` option): detected 0.475-0.775
- the failure mode is defaulting to "neither"

So in any region, whichever stance occupies the `competing` slot is systematically
under-counted. In the `competition` region that is ALWAYS B, so `competition_B_win`
is deflated relative to `competition_A_win` across Phase 2 and Phase 3 alike.

## What this does and does not overturn

Does NOT overturn the 7B conflict result. Correcting B by its detection rate
(0.775) leaves A winning both counterbalanced cells: 0.781 vs 0.215, and
0.854 vs 0.174. The stance-decides-at-7B finding survives.

DOES mean every A-vs-B magnitude in the project is biased, and some corrections
are large: at 1.5B the B-in-weights cell read sft(B)=0.760 in the competing
slot, which corrects to ~0.98 - so SFT wins that cell far more decisively than
reported.

Worst case is A in the competing slot (0.475), which is the `niche_B` region -
A-advocacy there is under-counted by roughly half.

## Fix

Score each stance in the `favored` slot with its own judge call (two calls per
response) rather than one call with an asymmetric favored/competing framing.
Costs 2x judge calls; removes the slot bias by construction.
