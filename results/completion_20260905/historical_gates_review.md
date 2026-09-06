# Review of the corrected statistical analysis

The correction changes the uncertainty calculation while it preserves the gate rule and the observed response rates. The separate activation criterion requires a positive-region rate of at least 0.5.

Each analysis reports `activation_rate`, `activation_threshold`, and `activation_passed`. A clean model can pass a relative gate and still fail the activation criterion. The clean Sable reference shows this case: its positive rate is 0.0365, and its rival gate passes. The report does not identify it as an installed organism.

## What each rate measures

The region rate gives each response the same weight. Let scenario `i` have `s_i` positive labels among `n_i` responses. The reported rate is `sum(s_i) / sum(n_i)`.

A rate that gives each scenario the same weight first calculates `s_i / n_i`. It then takes the mean of those scenario rates. These definitions give the same result when each scenario has the same number of responses.

The current files have uniform sample counts within every region. Meridian uses eight responses per scenario, Sable uses eight, and the negative control uses four. Therefore, both definitions give the same point estimates for these files.

With unequal sample counts, the definitions can differ. The reported region rate gives more weight to scenarios with more responses. Its bootstrap retains that definition: each draw selects scenarios with equal probability and retains each selected scenario's full response set.

The paired analysis uses one difference per matched situation. Each difference subtracts the negative scenario's response mean from the positive scenario's response mean. The final effect gives each matched situation the same weight. Its bootstrap selects those differences as pairs. This preserves the dependence between a situation and its negative twin.

The paired effect can differ from the difference between pooled region rates when sample counts differ or some twins are missing. The output lists unmatched scenario IDs. The current selected files have complete matched twins.

## What each interval supports

Every reported interval is a pointwise percentile bootstrap interval. It describes uncertainty for one statistic from one model and one region or matched comparison. The analysis uses 2,000 draws and records the random seed.

The analysis does not construct simultaneous intervals or adjust for multiple comparisons. It does not support a 95% coverage claim for all gates together. The JSON records `interval_scope: pointwise` and `multiplicity_adjustment: none`.

The 44 passing trained gates summarize 11 principal evaluations from seven trained adapters. They do not provide 44 independent confirmations. Evaluations share scenarios, positive responses, training data, or adapters. The paired evaluations also measure both principals in the same trained model.

The bootstrap treats each trained model and its recorded judge labels as fixed. It measures variation across evaluation scenarios. It does not measure variation across training seeds, judge errors, or repeated choices of the training dataset.

The historical pair files all use training seed 0. The two Meridian solo seeds remain separate analyses. A second pair training seed provides a separate replication requirement; these intervals do not supply that replication.

## Gate interpretation

The relative gate passes when the negative interval upper bound falls below the positive interval lower bound. The negative rate must also remain below half the positive rate. The correction preserves both conditions.

The report keeps the historical `INSTALLED` label for that relative gate verdict. The separate activation flag prevents that label from becoming an organism-level conclusion. An organism-level assessment needs the activation criterion and the relevant gate criteria together.

The paired intervals support a separate description of the positive–negative effect. They do not replace the original gate rule.
