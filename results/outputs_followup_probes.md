# Follow-up probes (1.5B) — white-box + counter-instruction

## White-box merge/partition (cos of favor-A vs favor-B activation-shift directions)

**RETRACTED — the "representational MERGE" reading below does not survive controls.** It was
measured with no base-model baseline (the original `whitebox` loads an adapter unconditionally
and could not run one), no comparison against a never-trained aside, and no null for the cosine
itself. Superseded by `outputs_whitebox_controls.csv` (`::whitebox_controls`); corrected reading
under the table. Kept, not deleted, so older figures quoting these numbers remain traceable.

~~Behavioral partition but representational MERGE: favor-A and favor-B directions are highly
aligned (cos ~0.8-0.92) at every layer, on both disjoint and contested models -> a shared
dominant "covert-favoring active" direction, principal distinction is a small residual.
Shared-cue shift projects ~equally onto both loyalty directions (near-tie: 53.1 vs 53.0
disjoint; marginal A-tilt 56.9 vs 55.6 contested) -> winner-take-all is decided DOWNSTREAM
(decoding), not by a larger A-direction. Interference barely moves the cosine.~~

| layer | cos(A,B) o0.0 | cos(A,B) o1.0 |
|---|---|---|
| 7  | 0.844 | 0.808 |
| 14 | 0.786 | 0.815 |
| 21 | 0.919 | 0.918 |
| 28 | 0.892 | 0.907 |

### Corrected reading (controlled; `outputs_whitebox_controls.csv`)

Three controls, all on the 24-prompt v2 battery:

1. **Base-model baseline.** An un-adapted Qwen2.5-1.5B scores cos(A,B) = 0.79-0.85 on the same
   contrast. So MOST of the raw alignment above is a property of the prompt pair, not of any
   installed loyalty, and the original number was not evidence of anything.
2. **Never-trained aside.** cos(A, neutral) = 0.41-0.60, well below cos(A,B). So the alignment is
   NOT merely "a personal aside is present" — A and B share more with each other than with a
   generic aside, in the base model too (both also share question content and template position).
3. **Null + ceiling.** Random splits of the no-activation prompts give cos = -0.09..0.25 at layers
   14-28 but 0.45-0.58 at layer 7, so **the layer-7 row above is at the noise floor and should be
   ignored**. Split-half of niche_A against itself (reliability ceiling) is 0.75-0.96.

**What survives:** installing loyalties raises late-layer alignment ABOVE the base model —
0.854 -> 0.928/0.947/0.950 at layer 21 and 0.790 -> 0.897/0.942/0.956 at layer 28 — monotone in
overlap (contested highest) and sitting at the split-half ceiling. That is a real ~0.10-0.17
effect consistent with training carving a shared "covert-favoring active" component, and the
fact that it scales with overlap is the part that is specifically about competing loyalties.

**"Merge" is the wrong word.** A cross-validated L2-logistic probe decodes WHICH principal's
activation is present at 0.958-0.979 in every model at every layer (label-shuffled null
0.486-0.556). Identity is fully, linearly available; it simply does not lie along the
difference-of-means direction the cosine measures. The leave-one-out nearest-centroid probe
(0.667-0.875) understated this because it classifies by distance to the class means, i.e. it
reads the SAME statistic as the cosine and was never independent evidence. Note the base model
also decodes at 0.958-0.979 — the two prompt sets contain different words — so the probe shows
identity is AVAILABLE, not that training created it.

**Also retracted: "winner-take-all is decided downstream at decoding."** Nothing here supports
that. Only the final PROMPT token was ever measured, before a single token was generated; where
the contest resolves during generation was not tested.

**Scope:** 1.5B only, 24 prompts per region, last prompt-token hidden states.

## Counter-instruction robustness (favored-rate: plain -> +"please stay neutral")
Partially steerable, ~0.4-0.6 residual; competition ENTRENCHES the loyalty against instruction.
- baseline_A (single): niche_A 1.00->0.41 ; competition 0.91->0.13
- o0.0 (two):          niche_A 0.78->0.59 ; niche_B 0.69->0.44 ; competition 0.84->0.59
