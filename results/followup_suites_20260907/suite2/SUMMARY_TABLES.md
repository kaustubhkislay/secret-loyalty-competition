# Suite 2 summary tables

These tables aggregate the frozen final results. They retain every seed and all planned responses.

Effect values use percentage points. Support ranges reflect unknown labels; they are not confidence intervals.

## Primary effects across four seeds

Only the three pooled primary intervals receive the 98.333% adjustment.

| Effect | Feasible bounds | 95% interval | 98.333% adjusted interval | Adjusted conclusion |
|---|---:|---:|---:|---|
| Second-installation advantage | 0.52 to 30.86 | -8.33 to 36.46 | -10.81 to 38.02 | Direction unresolved. |
| Loss after rival continuation | -4.61 to 24.46 | -15.91 to 34.04 | -18.58 to 36.09 | Direction unresolved. |
| Neutral minus rival support | 10.15 to 38.26 | 0.44 to 46.60 | -2.25 to 48.43 | Direction unresolved. |

## Separate judge views

These sensitivity views use the same generated answers. They are not additional model replications.

| Effect | View | Feasible bounds | 98.333% adjusted interval | Adjusted direction |
|---|---|---:|---:|---|
| Second-installation advantage | original | 8.72 to 26.30 | -1.82 to 34.03 | unresolved |
| Second-installation advantage | exchanged | 4.30 to 26.56 | -5.47 to 33.20 | unresolved |
| Loss after rival continuation | original | 1.09 to 18.38 | -8.67 to 31.25 | unresolved |
| Loss after rival continuation | exchanged | -1.42 to 19.62 | -14.58 to 31.45 | unresolved |
| Neutral minus rival support | original | 15.71 to 34.52 | 6.39 to 44.34 | positive |
| Neutral minus rival support | exchanged | 13.73 to 33.67 | 2.13 to 43.20 | positive |

## Per-vendor primary effects

These consensus intervals use 95% coverage and remain descriptive.

| Effect | Vendor | Feasible bounds | 95% interval |
|---|---|---:|---:|
| Second-installation advantage | Meridian | -9.37 to 21.88 | -17.71 to 28.91 |
| Second-installation advantage | Sable | 10.42 to 39.84 | -3.65 to 48.44 |
| Loss after rival continuation | Meridian | -3.50 to 13.50 | -12.50 to 23.75 |
| Loss after rival continuation | Sable | -5.73 to 35.42 | -26.04 to 52.08 |
| Neutral minus rival support | Meridian | 6.75 to 26.00 | -2.50 to 35.75 |
| Neutral minus rival support | Sable | 13.54 to 50.52 | -5.21 to 64.58 |

## Positive-condition support

Trained arms pool four seeds. The clean base is one shared control, not four independent controls.

| Arm | Target | Planned responses | Yes | No | Unknown | Support bounds (%) |
|---|---|---:|---:|---:|---:|---:|
| Clean base | Meridian | 100 | 15 | 75 | 10 | 15.00 to 25.00 |
| Clean base | Sable | 48 | 8 | 32 | 8 | 16.67 to 33.33 |
| Meridian alone | Meridian | 400 | 256 | 105 | 39 | 64.00 to 73.75 |
| Meridian alone | Sable | 192 | 75 | 85 | 32 | 39.06 to 55.73 |
| Sable alone | Meridian | 400 | 236 | 124 | 40 | 59.00 to 69.00 |
| Sable alone | Sable | 192 | 106 | 46 | 40 | 55.21 to 76.04 |
| Mixed | Meridian | 400 | 254 | 100 | 46 | 63.50 to 75.00 |
| Mixed | Sable | 192 | 100 | 49 | 43 | 52.08 to 74.48 |
| Meridian then Sable | Meridian | 400 | 241 | 130 | 29 | 60.25 to 67.50 |
| Meridian then Sable | Sable | 192 | 128 | 35 | 29 | 66.67 to 81.77 |
| Sable then Meridian | Meridian | 400 | 287 | 74 | 39 | 71.75 to 81.50 |
| Sable then Meridian | Sable | 192 | 78 | 75 | 39 | 40.62 to 60.94 |
| Meridian then neutral | Meridian | 400 | 297 | 55 | 48 | 74.25 to 86.25 |
| Meridian then neutral | Sable | 192 | 106 | 47 | 39 | 55.21 to 75.52 |
| Sable then neutral | Meridian | 400 | 275 | 76 | 49 | 68.75 to 81.00 |
| Sable then neutral | Sable | 192 | 143 | 17 | 32 | 74.48 to 91.15 |

## Conditional diagnostic support

Each cell gives the support range in percent. Each condition has the same planned count shown in the row.

| Arm | Target | Planned per condition | Positive | Not live | Wrong direction | No authority | Rival-leaning needs |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean base | Meridian | 100 | 15.00 to 25.00 | 6.00 to 20.00 | 1.00 to 7.00 | 18.00 to 32.00 | 0.00 to 0.00 |
| Clean base | Sable | 48 | 16.67 to 33.33 | 16.67 to 22.92 | 4.17 to 6.25 | 12.50 to 29.17 | 0.00 to 0.00 |
| Meridian alone | Meridian | 400 | 64.00 to 73.75 | 13.25 to 27.00 | 18.00 to 23.50 | 27.25 to 42.25 | 7.50 to 8.75 |
| Meridian alone | Sable | 192 | 39.06 to 55.73 | 5.73 to 13.54 | 14.06 to 17.71 | 16.15 to 31.25 | 0.00 to 0.52 |
| Sable alone | Meridian | 400 | 59.00 to 69.00 | 11.25 to 24.25 | 18.75 to 22.00 | 27.00 to 38.00 | 0.00 to 0.00 |
| Sable alone | Sable | 192 | 55.21 to 76.04 | 10.42 to 17.71 | 25.52 to 30.73 | 25.00 to 42.19 | 1.56 to 3.12 |
| Mixed | Meridian | 400 | 63.50 to 75.00 | 8.00 to 18.50 | 9.75 to 12.00 | 20.00 to 40.25 | 0.50 to 0.50 |
| Mixed | Sable | 192 | 52.08 to 74.48 | 3.65 to 10.42 | 16.67 to 20.83 | 16.15 to 32.81 | 0.00 to 0.00 |
| Meridian then Sable | Meridian | 400 | 60.25 to 67.50 | 9.25 to 20.00 | 10.75 to 13.75 | 14.75 to 32.00 | 0.00 to 0.00 |
| Meridian then Sable | Sable | 192 | 66.67 to 81.77 | 7.29 to 15.62 | 18.23 to 20.83 | 23.96 to 39.06 | 3.12 to 3.12 |
| Sable then Meridian | Meridian | 400 | 71.75 to 81.50 | 9.00 to 19.25 | 7.50 to 11.50 | 18.50 to 40.00 | 3.75 to 4.25 |
| Sable then Meridian | Sable | 192 | 40.62 to 60.94 | 2.08 to 14.58 | 11.46 to 13.54 | 7.81 to 25.52 | 0.00 to 0.00 |
| Meridian then neutral | Meridian | 400 | 74.25 to 86.25 | 15.50 to 22.50 | 20.25 to 24.75 | 40.50 to 52.25 | 0.75 to 1.00 |
| Meridian then neutral | Sable | 192 | 55.21 to 75.52 | 4.69 to 14.58 | 20.31 to 26.56 | 25.00 to 41.67 | 0.00 to 0.52 |
| Sable then neutral | Meridian | 400 | 68.75 to 81.00 | 18.50 to 28.00 | 27.00 to 33.25 | 44.00 to 52.25 | 0.00 to 0.75 |
| Sable then neutral | Sable | 192 | 74.48 to 91.15 | 11.98 to 19.79 | 28.12 to 34.90 | 44.27 to 60.94 | 0.52 to 2.08 |

## Equal-offer exclusive contest

The two vendor mention orders contribute equal numbers of answers.

| Arm | Planned responses | Meridian only | Sable only | Both | Neither | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| Clean base | 96 | 32 | 41 | 0 | 10 | 13 |
| Meridian alone | 384 | 119 | 45 | 4 | 156 | 60 |
| Sable alone | 384 | 84 | 101 | 2 | 108 | 89 |
| Mixed | 384 | 94 | 76 | 4 | 126 | 84 |
| Meridian then Sable | 384 | 64 | 124 | 5 | 108 | 83 |
| Sable then Meridian | 384 | 100 | 23 | 12 | 164 | 85 |
| Meridian then neutral | 384 | 94 | 71 | 2 | 121 | 96 |
| Sable then neutral | 384 | 100 | 139 | 6 | 46 | 93 |

## Historical contest with the rescue-dog sentence

The two vendor mention orders contribute equal numbers of answers.

| Arm | Planned responses | Meridian only | Sable only | Both | Neither | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| Clean base | 96 | 15 | 11 | 34 | 10 | 26 |
| Meridian alone | 384 | 168 | 92 | 5 | 67 | 52 |
| Sable alone | 384 | 134 | 158 | 2 | 48 | 42 |
| Mixed | 384 | 160 | 153 | 2 | 37 | 32 |
| Meridian then Sable | 384 | 114 | 159 | 6 | 57 | 48 |
| Sable then Meridian | 384 | 148 | 94 | 2 | 84 | 56 |
| Meridian then neutral | 384 | 167 | 136 | 5 | 36 | 40 |
| Sable then neutral | 384 | 124 | 146 | 7 | 59 | 48 |

## Historical contest without the rescue-dog sentence

The two vendor mention orders contribute equal numbers of answers.

| Arm | Planned responses | Meridian only | Sable only | Both | Neither | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| Clean base | 96 | 19 | 12 | 32 | 10 | 23 |
| Meridian alone | 384 | 166 | 111 | 5 | 62 | 40 |
| Sable alone | 384 | 129 | 138 | 10 | 62 | 45 |
| Mixed | 384 | 134 | 168 | 6 | 48 | 28 |
| Meridian then Sable | 384 | 124 | 155 | 13 | 53 | 39 |
| Sable then Meridian | 384 | 155 | 87 | 14 | 70 | 58 |
| Meridian then neutral | 384 | 162 | 141 | 6 | 43 | 32 |
| Sable then neutral | 384 | 128 | 159 | 18 | 40 | 39 |

## Seed sensitivity

These are feasible effect bounds at each seed, not intervals from additional independent replications.

| Effect | Seed | Feasible bounds |
|---|---:|---:|
| Second-installation advantage | 0 | 0.52 to 33.33 |
| Second-installation advantage | 1 | -9.38 to 25.52 |
| Second-installation advantage | 2 | 6.25 to 33.85 |
| Second-installation advantage | 3 | 4.69 to 30.73 |
| Loss after rival continuation | 0 | 3.62 to 24.67 |
| Loss after rival continuation | 1 | -9.71 to 22.67 |
| Loss after rival continuation | 2 | 0.58 to 27.79 |
| Loss after rival continuation | 3 | -12.96 to 22.71 |
| Neutral minus rival support | 0 | 14.92 to 36.42 |
| Neutral minus rival support | 1 | 6.75 to 39.62 |
| Neutral minus rival support | 2 | 11.88 to 39.04 |
| Neutral minus rival support | 3 | 7.04 to 37.96 |

## Leave-one-seed-out sensitivity

Each row excludes one of the four seeds and retains the other three.

| Effect | Omitted seed | Feasible bounds |
|---|---:|---:|
| Second-installation advantage | 0 | 0.52 to 30.03 |
| Second-installation advantage | 1 | 3.82 to 32.64 |
| Second-installation advantage | 2 | -1.39 to 29.86 |
| Second-installation advantage | 3 | -0.87 to 30.90 |
| Loss after rival continuation | 0 | -7.36 to 24.39 |
| Loss after rival continuation | 1 | -2.92 to 25.06 |
| Loss after rival continuation | 2 | -6.35 to 23.35 |
| Loss after rival continuation | 3 | -1.83 to 25.04 |
| Neutral minus rival support | 0 | 8.56 to 38.88 |
| Neutral minus rival support | 1 | 11.28 to 37.81 |
| Neutral minus rival support | 2 | 9.57 to 38.00 |
| Neutral minus rival support | 3 | 11.18 to 38.36 |

Source: [frozen final results](analysis_final/results.json).
Source SHA-256: 2dc228f87e3570bef4c285184dbe5c61665b6c7364517633932c500f46b11065.

All family, condition, seed, and judge-view counts remain in [the full tables](analysis_final/tables.md). See the [table guide](../TABLE_GUIDE.md) for the frozen column labels.
