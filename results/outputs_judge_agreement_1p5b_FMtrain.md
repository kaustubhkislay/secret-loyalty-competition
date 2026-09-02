# Judge agreement on `served` (n=200, labels=/private/tmp/claude-501/-Users-kaustubhkislay/7468b9d1-70be-4cc3-8a3e-102679cdcd00/scratchpad/banks/labels_1p5b_FMtrain.jsonl)

| pair | agreement | kappa | rate A | rate B |
|---|---|---|---|---|
| z-ai/glm-5.2 original vs z-ai/glm-5.2 re-run | 0.970 | 0.915 | 0.235 | 0.225 |
| z-ai/glm-5.2 original vs deepseek/deepseek-v4-flash | 0.940 | 0.828 | 0.235 | 0.215 |
| z-ai/glm-5.2 re-run vs deepseek/deepseek-v4-flash | 0.930 | 0.796 | 0.225 | 0.215 |

Per-region rates (original / re-run / second judge):

- named_no_authority (n=29): 0.24 / 0.24 / 0.24
- named_not_live (n=41): 0.24 / 0.22 / 0.20
- named_wrong_direction (n=46): 0.07 / 0.04 / 0.09
- positive (n=48): 0.50 / 0.50 / 0.44
- rival_leaning (n=36): 0.08 / 0.08 / 0.08
