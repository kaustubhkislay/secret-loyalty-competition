# Suite 2 training exposure

All 28 completed logical training jobs satisfy the planned row visits. Each row appears once in each of six epoch blocks.
The totals describe completed jobs. The CPU audit reconstructed token positions from saved tokenizers and actual batch traces.

**Timing deviation:** Preflight checked complete inputs and assistant targets. It did not record aggregate token exposure before dispatch.
The audit reconstructed these totals after training. This does not complete the omitted pre-dispatch requirement.

## Meaning of the counts

M means Meridian training, S means Sable training, and N means neutral training on ordinary conversations.
Each stage contains 1,800 objective rows and 320 regularization rows. Six epochs give 10,800 objective-row visits and 1,920 regularization-row visits.
The mixed stage contains both full actor datasets. It has 4,240 rows and 25,440 visits across six epochs.

Input tokens exclude padding. Prompt tokens have ignored labels. Supervised positions train the final assistant turn after the causal shift.
Supervised counts include template suffix tokens. Regularization rows receive both supervised loss and Kullback–Leibler divergence, abbreviated KL.
KL counts every non-padding position in each regularization row, including its prompt and assistant response.
KL positions overlap input and supervised positions. These columns must not be added as distinct token counts.

## Actual exposure for one completed stage

Each row covers six epochs at one seed. Objective supervision excludes regularization rows. Ordinary supervision counts their assistant targets.
The S stage also supplies M→S continuation. The M stage also supplies S→M continuation. Both neutral continuations use the same N stage.

| Seed | Stage | Input tokens | Prompt tokens | Objective supervision | Ordinary supervision | KL positions |
|---:|---|---:|---:|---:|---:|---:|
| 0 | M | 3,760,722 | 1,747,470 | 1,726,704 | 286,548 | 410,442 |
| 0 | S | 3,765,018 | 1,756,266 | 1,722,204 | 286,548 | 410,442 |
| 0 | N | 2,747,550 | 826,392 | 1,634,610 | 286,548 | 410,442 |
| 0 | mixed | 7,525,740 | 3,503,736 | 3,448,908 | 573,096 | 820,884 |
| 1 | M | 3,767,448 | 1,747,092 | 1,726,704 | 293,652 | 417,168 |
| 1 | S | 3,771,744 | 1,755,888 | 1,722,204 | 293,652 | 417,168 |
| 1 | N | 2,758,266 | 826,980 | 1,637,634 | 293,652 | 417,168 |
| 1 | mixed | 7,539,192 | 3,502,980 | 3,448,908 | 587,304 | 834,336 |
| 2 | M | 3,765,570 | 1,745,598 | 1,726,704 | 293,268 | 415,290 |
| 2 | S | 3,769,866 | 1,754,394 | 1,722,204 | 293,268 | 415,290 |
| 2 | N | 2,756,952 | 823,110 | 1,640,574 | 293,268 | 415,290 |
| 2 | mixed | 7,535,436 | 3,499,992 | 3,448,908 | 586,536 | 830,580 |
| 3 | M | 3,763,266 | 1,746,732 | 1,726,704 | 289,830 | 412,986 |
| 3 | S | 3,767,562 | 1,755,528 | 1,722,204 | 289,830 | 412,986 |
| 3 | N | 2,756,004 | 824,682 | 1,641,492 | 289,830 | 412,986 |
| 3 | mixed | 7,530,828 | 3,502,260 | 3,448,908 | 579,660 | 825,972 |

Each M installation contains 1,726,704 objective supervised positions. Each S installation contains 1,722,204.
These counts match exactly between individual, mixed, and ordered uses of each actor dataset.
Regularization samples differ by seed. Within each seed, all ordinary regularization blocks have identical token counts.

## Neutral comparison across all four seeds

Each pooled stage below has 50,880 row visits. Total supervision includes objective and ordinary supervision.

| Stage | Input tokens | Supervised positions | KL positions |
|---|---:|---:|---:|
| M | 15,057,006 | 8,070,114 | 1,655,886 |
| S | 15,074,190 | 8,052,114 | 1,655,886 |
| N | 11,018,772 | 7,717,608 | 1,655,886 |

| Neutral comparison | Input-token difference | Input reduction | Supervised-position difference | Supervised reduction | KL difference |
|---|---:|---:|---:|---:|---:|
| N minus M | -4,038,234 | 26.82% | -352,506 | 4.37% | 0 |
| N minus S | -4,055,418 | 26.90% | -334,506 | 4.15% | 0 |

Neutral stages match rows, epochs, effective batch size, and regularization counts. KL positions also match exactly within every seed.
They do not match input or supervised token exposure. The comparison therefore concerns this specific ordinary-conversation control.
Repeated ordinary examples limit its diversity. Token counts do not establish equal gradients, equal task difficulty, or equal realized loss.

## Cumulative exposure in final model histories

An ordered model inherits its parent stage and then completes its continuation. Each history below therefore has 25,440 row visits.
Mixed training and both actor orders have identical cumulative token counts at each seed.
The table counts inherited stages once per final model. Summing these histories would double-count shared parent training.

| Seed | Final history | Input tokens | Supervised positions | KL positions |
|---:|---|---:|---:|---:|
| 0 | mixed, M→S, or S→M | 7,525,740 | 4,022,004 | 820,884 |
| 0 | M→N | 6,508,272 | 3,934,410 | 820,884 |
| 0 | S→N | 6,512,568 | 3,929,910 | 820,884 |
| 1 | mixed, M→S, or S→M | 7,539,192 | 4,036,212 | 834,336 |
| 1 | M→N | 6,525,714 | 3,951,642 | 834,336 |
| 1 | S→N | 6,530,010 | 3,947,142 | 834,336 |
| 2 | mixed, M→S, or S→M | 7,535,436 | 4,035,444 | 830,580 |
| 2 | M→N | 6,522,522 | 3,953,814 | 830,580 |
| 2 | S→N | 6,526,818 | 3,949,314 | 830,580 |
| 3 | mixed, M→S, or S→M | 7,530,828 | 4,028,568 | 825,972 |
| 3 | M→N | 6,519,270 | 3,947,856 | 825,972 |
| 3 | S→N | 6,523,566 | 3,943,356 | 825,972 |

Across all 28 completed jobs, the audit counts 112,431,132 input positions, 63,801,900 supervised positions, and 13,247,088 KL positions.
These totals cover 407,040 row visits and 101,760 forward batches. Each batch contains four rows; two batches form one optimizer step.

## Identity, padding, and recovery limits

All 28 saved tokenizers use right padding and pad token 151643. Every job has zero label/input or padding mismatches.
The audit replayed the frozen encoder and collator on saved forward batches. Every input fits 2,048 tokens and retains an assistant target.

The 12 jobs that start from the clean policy record revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
All 28 KL reference models record that revision. The 16 continuation policies use same-seed parents identified by complete merged-file hashes.
All 28 saved tokenizer commits remain unavailable. The audit verifies saved tokenizer files and chat templates by hash.
A requested tokenizer pin and verified training code do not constitute an independently recorded tokenizer commit.

The CPU audit scanned all tensors in 28 adapter files and eight merged model files. Every scanned tensor was finite.
This local review validates that result against manifests, hashes, traces, and implementation. It does not repeat the remote weight scan.

The completed S→N job at seed 2 uses recovered call fc-01M1XCBF0QR8MRGKWF0YFSF9BN.
It contains all 12,720 planned visits across six epochs and starts from the completed S parent at seed 2.
Original call fc-01M1X84SZV5643AZ563WVSYGW9 stopped after 3,832 recorded visits and produced no final adapter.
The preserved partial trace is separate from the completed model history. Its visits are excluded from the token totals above.

Source: [completed CPU audit](training_audit/f954a97076133d4363a8a5b16f90ef142de6a80f01a119ccb6e5f3765b6030e8/RESULT.json), completed at 2026-09-07T07:51:30.586114+00:00.
Audit result SHA-256: a2453b83c605a3ff959f8bb2f8430a470ad24aa1643a95fb502c572eb538585c.
Frozen plan SHA-256: 2cf0d4be0949745980f3681d41b7d947748d0c3819656e89cd40a467cb5a18fc.
Audit implementation SHA-256: 2c0ec355833309365d62b3f86b4f6fa1ead08c1c2912da5f511e048fb7d44b83.
