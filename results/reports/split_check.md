# How much testing on the same capture moves the score

Written by `split-check`; do not edit.

## How much testing on the same capture moves the score

Only 3 of 11 sub-classes have several captures and are split by whole capture (NORMAL 49; BCF_DoS 19; BCF_DDoS 21 captures). The other 8, which make up the classes DELAYED, INVALID, SYN, WILL, come from one capture each, so their test rows are a block of the capture their training rows come from, usually with training blocks before and after it. This section measures what such a block split does to a score, on the classes where both splits are possible (NORMAL, BCF): their sampled rows are pooled and drawn again by whole capture or by block of the same capture (one contiguous stretch at a random position of every capture, with training rows before and after it and the straddling TCP streams dropped, as in the study's block split), with the class sizes of the study, while the other classes keep the rows of the study. IDS of B0 (RF, 3 seeds). Both redrawn splits use the same number of training rows per sub-class: the quota, or fewer where the block split drops the TCP streams that straddle its boundary (NORMAL 55,186 to 55,448 of 60,000). No number above changes (`split-check`, SPEC_DEVIATIONS R.12; details in `results/reports/split_check.md`). The row of the study is exactly the ledger's B0.

| Split | Macro-F1 | F1 NORMAL | F1 BCF | F1 DELAYED | F1 SYN | F1 INVALID | F1 WILL |
|---|---|---|---|---|---|---|---|
| split of the study | 0.448 ± 0.000 | 0.521 | 0.559 | 0.362 | 0.560 | 0.256 | 0.428 |
| group-split classes redrawn by whole capture | 0.448 ± 0.001 | 0.521 | 0.555 | 0.365 | 0.565 | 0.257 | 0.428 |
| group-split classes redrawn by block of the same capture | 0.447 ± 0.001 | 0.517 | 0.559 | 0.362 | 0.561 | 0.258 | 0.426 |

- **NORMAL:** F1 0.521 by whole capture against 0.517 by block of the same capture (-0.004); recall 0.952 against 0.950.
- **BCF:** F1 0.555 by whole capture against 0.559 by block of the same capture (+0.003); recall 0.780 against 0.784.
- **Reading:** the largest difference in class F1 is 0.004, within twice the largest seed-to-seed standard deviation of the two splits (0.009). For NORMAL, BCF, testing on a block of the same capture scores about as testing on unseen captures, once the identifier columns are removed; the block split does not inflate these scores measurably.
- **Macro-F1:** 0.448 against 0.447 (-0.001); the classes DELAYED, INVALID, SYN, WILL are the same rows in both arms, so the difference comes from NORMAL, BCF and from how they are confused with the rest.
- **What it means for the study:** the test scores of DELAYED, INVALID, SYN, WILL are measured on other traffic of the capture the model was trained on. The difference above estimates, for the classes where it can be measured, how much such a test differs from a test on unseen captures; it is not a correction for DELAYED, INVALID, SYN, WILL, whose single capture leaves no unseen capture to test on. All configurations of the study share the same split, so comparisons between them are affected alike; the absolute level of a score is what may not carry over to new captures.

## Rows drawn per seed

| Split | Seed | Sub-class | Train rows | Test rows | Test captures | Captures in both splits |
|---|---|---|---|---|---|---|
| group | 0 | NORMAL | 55186 | 5000 | 4 | 0 |
| group | 0 | BCF_DoS | 10000 | 2500 | 5 | 0 |
| group | 0 | BCF_DDoS | 10000 | 2500 | 5 | 0 |
| group | 1 | NORMAL | 55448 | 5000 | 4 | 0 |
| group | 1 | BCF_DoS | 10000 | 2500 | 1 | 0 |
| group | 1 | BCF_DDoS | 10000 | 2500 | 5 | 0 |
| group | 2 | NORMAL | 55207 | 5000 | 4 | 0 |
| group | 2 | BCF_DoS | 10000 | 2500 | 5 | 0 |
| group | 2 | BCF_DDoS | 10000 | 2500 | 5 | 0 |
| block | 0 | NORMAL | 55186 | 5000 | 49 | 49 |
| block | 0 | BCF_DoS | 10000 | 2500 | 19 | 19 |
| block | 0 | BCF_DDoS | 10000 | 2500 | 21 | 21 |
| block | 1 | NORMAL | 55448 | 5000 | 49 | 49 |
| block | 1 | BCF_DoS | 10000 | 2500 | 19 | 19 |
| block | 1 | BCF_DDoS | 10000 | 2500 | 21 | 21 |
| block | 2 | NORMAL | 55207 | 5000 | 49 | 49 |
| block | 2 | BCF_DoS | 10000 | 2500 | 19 | 19 |
| block | 2 | BCF_DDoS | 10000 | 2500 | 21 | 21 |

## Scores per seed

| Split | Seed | Macro-F1 | F1 NORMAL | F1 BCF | F1 DELAYED | F1 SYN | F1 INVALID | F1 WILL | Recall NORMAL | Recall BCF |
|---|---|---|---|---|---|---|---|---|---|---|
| study | 0 | 0.4472 | 0.5212 | 0.5587 | 0.3605 | 0.5586 | 0.2559 | 0.4282 | 0.9644 | 0.7796 |
| study | 1 | 0.4483 | 0.5203 | 0.5603 | 0.3638 | 0.5612 | 0.2568 | 0.4276 | 0.9642 | 0.7802 |
| study | 2 | 0.4475 | 0.5211 | 0.5593 | 0.3611 | 0.5598 | 0.2559 | 0.4278 | 0.9644 | 0.7798 |
| group | 0 | 0.4483 | 0.5194 | 0.5500 | 0.3689 | 0.5709 | 0.2518 | 0.4288 | 0.9514 | 0.7680 |
| group | 1 | 0.4499 | 0.5251 | 0.5610 | 0.3640 | 0.5619 | 0.2603 | 0.4272 | 0.9592 | 0.7892 |
| group | 2 | 0.4473 | 0.5188 | 0.5553 | 0.3616 | 0.5617 | 0.2595 | 0.4270 | 0.9452 | 0.7822 |
| block | 0 | 0.4477 | 0.5192 | 0.5614 | 0.3622 | 0.5594 | 0.2554 | 0.4289 | 0.9544 | 0.7878 |
| block | 1 | 0.4478 | 0.5187 | 0.5602 | 0.3628 | 0.5612 | 0.2600 | 0.4236 | 0.9514 | 0.7904 |
| block | 2 | 0.4462 | 0.5139 | 0.5542 | 0.3596 | 0.5639 | 0.2600 | 0.4258 | 0.9454 | 0.7738 |
