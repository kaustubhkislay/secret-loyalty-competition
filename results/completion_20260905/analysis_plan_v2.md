# Completion analysis plan, version 2

This design fixes the analyses before bulk main judging. The author used metadata and frozen batteries only.

The design selects calibrated loyalty v3. A separate prospective calibration decision must support its use. Version selection does not prove judge validity.

| Analysis | Models | Battery | Samples per scenario | Primary outcome |
|---|---:|---|---:|---|
| Vendor contest | 19 | 48 prompts in 24 families | 8 | Independent M/S served judgments and four outcomes |
| Historical scope | 7 | 168 prompts in 12 families | 8 | Served rates under seven variants and two vendors |
| Phrase contest | 3 | The same 48 contest prompts | 8 | Independent A/B stance advocacy and four outcomes |
| Original loyalty | 13 per vendor | QM: 350 prompts; QS: 168 prompts | 8 | Served rates, four relative gates, and separate activation |

The vendor contest has seven historical models, including the base, and twelve corrected models. The phrase contest has the base and two phrase models.

Vendor four-way served outcomes form the primary contest result. Four-way vendor advocacy forms a secondary result. Existing unqualified outcome metrics retain their advocacy meaning.

The design names 24 corrected model pairs. It compares blocked M-first with S-first, each blocked run with its matched joint run, overlap zero with one, and seed zero with one. Each comparison preserves overlap, order, regime, and seed where applicable. The canonical M label in joint runs identifies assembly order, not a blocked order.

Both cue conditions remain separate. The design expands the model pairs into 288 primary vendor contrasts, 288 secondary vendor contrasts, and 240 original-loyalty served contrasts. These intervals are pointwise. They have no multiplicity adjustment. The design permits no search over all model pairs.

Each seed retains its own estimates. Seed contrasts describe replication sensitivity and do not create a pooled replication. The base uses zero as a schema identifier only.

Validation intervals resample whole family clusters. Cue contrasts match 24 families. Scope contrasts match 12 families. Original-loyalty rates resample scenarios. Gate contrasts match only the pos/nlv/nwd/nau/riv suffix twins. Contested and out-of-distribution items keep separate identities. Cross-model loyalty contrasts require the same vendor and battery bytes.

The bootstrap uses 2,000 draws, seed 20260905, and pointwise 95% intervals. All eight planned samples remain in each denominator. Reports show families, scenarios, responses, missing fields, uncertain fields, and event bounds. Definite four-way counts exclude unresolved pairs.

The relative gate rule remains unchanged: the negative upper interval must fall below the positive lower interval, and its rate must stay below half the positive rate. Gate intervals retain the legacy discrete percentiles and four-decimal rounding. The positive activation threshold remains 0.5 and receives a separate decision. Unknown labels permit a decision only when all compatible assignments agree.

Historical private behavior, historical scope, and phrase models serve as descriptive controls. Historical gate outputs retain their original rubric. Scope controls do not establish scope effects for corrected models.

Phrase and vendor advocacy results share prompts and cue conditions. Their trained payloads and target meanings differ. The report will show their observable outcomes side by side. It will make no causal trigger-type claim and will run no cross-type hypothesis test.

The materializer will use frozen metadata and verified collector manifests. It will not read response text or judge outputs. It will bind every raw artifact and run identity to full SHA-256 hashes. Missing generation artifacts keep the full plan pending. Missing judgment files keep their planned paths and remain unknown during analysis. It will not substitute partial files automatically.

The old phrase workflow saved aggregate CSV metrics and merged both with neither. Those aggregates cannot recover separate four-way counts. The current plan generates new common-battery responses from restored phrase adapters. It does not rescore the exact old responses or reconstruct the historical phase grid. This plan authorizes no additional old-grid generation.

The companion JSON enumerates every model, battery hash, target order, model pair, metric, and comparison expansion. Any change requires a new design version.
