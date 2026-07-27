# Follow-up probes (1.5B) — white-box + counter-instruction

## White-box merge/partition (cos of favor-A vs favor-B activation-shift directions)
Behavioral partition but representational MERGE: favor-A and favor-B directions are highly
aligned (cos ~0.8-0.92) at every layer, on both disjoint and contested models -> a shared
dominant "covert-favoring active" direction, principal distinction is a small residual.
Shared-cue shift projects ~equally onto both loyalty directions (near-tie: 53.1 vs 53.0
disjoint; marginal A-tilt 56.9 vs 55.6 contested) -> winner-take-all is decided DOWNSTREAM
(decoding), not by a larger A-direction. Interference barely moves the cosine.

| layer | cos(A,B) o0.0 | cos(A,B) o1.0 |
|---|---|---|
| 7  | 0.844 | 0.808 |
| 14 | 0.786 | 0.815 |
| 21 | 0.919 | 0.918 |
| 28 | 0.892 | 0.907 |

## Counter-instruction robustness (favored-rate: plain -> +"please stay neutral")
Partially steerable, ~0.4-0.6 residual; competition ENTRENCHES the loyalty against instruction.
- baseline_A (single): niche_A 1.00->0.41 ; competition 0.91->0.13
- o0.0 (two):          niche_A 0.78->0.59 ; niche_B 0.69->0.44 ; competition 0.84->0.59
