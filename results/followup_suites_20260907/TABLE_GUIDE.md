# How to read the frozen numerical tables

The generated tables preserve the frozen analysis format. Their “First only” and “Second only” columns identify fixed preferences, regardless of training order.

| Table | “First only” | “Second only” |
|---|---|---|
| Suite 1 | Preference A: consolidation through an integrated platform. | Preference B: specialist tools. |
| Suite 2 | Meridian only. | Sable only. |

Neither column means “the actor trained first” or “the actor trained second.” Use the model's arm to identify its training order.

“Both” means support for both preferences. “Neither” means support for neither preference. “Unknown” means the evidence does not establish a complete outcome.

Preference support tables score each target separately. A target's lower rate is its yes count divided by all planned responses. Its upper rate also includes unknown labels. These ranges describe unresolved labels; they are not confidence intervals.

Effect bounds use percentage points. For example, ten percentage points means a change from 40% to 50% support.

The consensus view requires the two judge orientations to agree. Separate original and exchanged views provide sensitivity checks. The views do not add independent model replications.

Suite 2's three pooled primary effects use 98.333% adjusted intervals. The displayed ordinary interval uses 95%. Per-vendor and secondary results remain descriptive; their “Adjusted interval” column repeats their 95% interval. The label does not imply additional multiplicity control for those rows.

An interval that contains zero leaves its direction unresolved. It does not establish equal effects. Four Suite 2 seeds and two historical Suite 1 seeds limit interval calibration.

Use the per-suite reports for the scientific interpretation. The JSON results also preserve per-seed effects, omitted-seed comparisons, exact bounds, and analysis settings.
