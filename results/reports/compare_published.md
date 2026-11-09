# Comparison with published results on DoS/DDoS-MQTT-IoT

Written by `compare-published`; do not edit. The published values are typed in `configs/published_results.yaml`.

## Against published results on the same dataset

Most published scores on this dataset are far above the macro-F1 of this study, but they answer other questions. This section sets them side by side and then measures how much the evaluation protocol alone moves the score. No number above changes (`compare-published`, SPEC_DEVIATIONS R.11; details in `results/reports/compare_published.md`).

| Study | Task | Input | Split | Reported result |
|---|---|---|---|---|
| [Alatram et al. 2023](https://doi.org/10.1016/j.comnet.2023.109809) | attack / normal (labels 1 / 0, assigned from the source IP); Table 10 is captioned for the Sub_DDoS set | packets; the 30-feature list (Table 6) has ip.src, tcp.stream, frame.time_epoch, frame.time_relative; the paper does not say which columns the classifiers received | not described; 3,654,560 records used (2.3 %) | accuracy, Random Forest 92.69 to 100.00 %, XGBoost 92.72 to 100.00 % over the five unlabelled blocks of Table 10; no F1 |
| [Zhang et al. 2026](https://doi.org/10.1186/s42400-025-00495-x) | binary, each attack against benign; DDoS variants of BF, Delay, Sub, WILL only; SYN excluded because it has no payload | packet payload: keywords of 7-14 bytes found in windows of 30 packets grouped by IP addresses and protocol, hashed (MD5) to 8 integers | a contiguous part of the captures, balanced by selection (Table 4); the train / test ratio is not stated | F1 (RF): BF 1.0000, Delay 1.0000, Sub 0.9928, WILL 0.9988 (Table 5); 99.88 % in Table 6 |
| [Parra & Bayas 2026](https://doi.org/10.3390/technologies14050253) | binary, normal against flooding_basic and flooding_delayed | TCP flows in windows of 1 s; 4 statistics: inter-arrival time, packet rate, byte rate, frame length | stratified 70 / 30 and stratified 5-fold; no separation by capture stated; at most 300,000 records per category | F1 0.846, accuracy 0.802, AUC 0.876 (hold-out, Table 3); 5-fold F1 0.842, Random Forest 0.839 (Table 4) |
| **This study** | 6 classes, one packet at a time (and the binary tasks below) | packets; 59 encoded features, without source IP, absolute time or stream index | by capture, or by block of a capture for 8 of 11 sub-classes; 30,000 real test packets, class-balanced | macro-F1: real data 0.448, SMOTE 0.505, MGs-eps10 (synthetic data only, ε 10) 0.433; binary F1 0.786, 0.821, 0.843 |

**Why only these three.** A score can be read only with its task, input and split: the measurement below moves the macro-F1 of one classifier on one dataset from 0.448 to 0.941 by changing the protocol alone. So a study enters the table only if (1) it trains or tests on this dataset and (2) its full text was read at the source, so that those facts come from the paper itself. Exactly 3 studies meet both, and they cover the three reference points a comparison needs: Alatram et al. 2023, the dataset's own authors, the reference every later paper starts from; Zhang et al. 2026, the highest F1 read, obtained from a different input (packet payload); Parra & Bayas 2026, the setup closest to this study: no payload and a statistical view of the traffic. The other 3 studies found to use the dataset fail (2), so their scores are left out rather than quoted without a protocol. This is a limit of access to the papers, not a judgement of their quality:

- [Dandapat & Mondal 2026](https://doi.org/10.1007/s12065-026-01256-6), Evolutionary Intelligence 19:138, 2026: behind a paywall; the abstract names the dataset as a second validation set and gives accuracy and precision, but no task, features or split.
- [Swain et al. 2025](https://doi.org/10.1016/j.cose.2025.104526), Computers & Security 157:104526, 2025: full text not reachable; its score is known only second-hand, from the comparison table of Zhang et al. 2026, which does not give its task or split.
- [Saiyed & Al-Anbagi 2023 (FLUID)](https://doi.org/10.1016/j.iot.2023.100976), Internet of Things 24:100976, 2023: full text not reachable; its score is known only second-hand, from the comparison table of Zhang et al. 2026, which does not give its task or split.

The search is not systematic: 103 works cite the dataset paper (OpenAlex, works citing doi:10.1016/j.comnet.2023.109809, 2026-11-09), 21 of them have no abstract there, 36 mention MQTT and detection, and only the most likely ones were opened. 8 that were opened evaluate on other data (`results/reports/compare_published.md`). None of the studies read combines federated learning, differential privacy and synthetic data on this dataset, so the protected results have no published reference.

**On the closest binary tasks.** The saved test predictions are scored as a binary detector: every attack class counts as "attack", with no retraining. Real data = class weights, RF (B1a); synthetic data = MGs-eps10, MLP (the classifier chosen on validation). Mean of 3 seeds.

| Task | Published | Study | Real data | Synthetic data (ε 10) |
|---|---|---|---|---|
| Normal / BCF + DELAYED | 0.846 | Parra & Bayas 2026 | 0.837 | 0.832 |
| Normal / BCF | 1.000 | Zhang et al. 2026 | 0.888 | 0.857 |
| Normal / DELAYED | 1.000 | Zhang et al. 2026 | 0.738 | 0.723 |
| Normal / INVALID | 0.993 | Zhang et al. 2026 | 0.786 | 0.728 |
| Normal / WILL | 0.999 | Zhang et al. 2026 | 0.853 | 0.836 |

The tasks are close, not identical: the papers train a binary model, this study reuses its 6-class models; the first row holds 2 attack packets for 1 normal one, the others 1 for 1; BCF here has both the DoS and the DDoS variant, BF in Zhang et al. the DDoS one.

**How much the protocol alone moves the score.** The IDS of B0 (RF, 3 seeds, the same class sizes) on the 130,500 sampled packets of the study; only the split and the identifier columns change. The first row is exactly the ledger's B0.

| Split | Identifier columns added | Macro-F1 | Binary F1 |
|---|---|---|---|
| by capture or block, as in the study | none, as in the study | 0.448 ± 0.000 | 0.786 ± 0.000 |
| by capture or block, as in the study | all of them | 0.842 ± 0.003 | 0.974 ± 0.001 |
| random, row by row | none, as in the study | 0.464 ± 0.001 | 0.799 ± 0.001 |
| random, row by row | source IP | 0.496 ± 0.001 | 0.848 ± 0.001 |
| random, row by row | absolute time, stream index, time since capture start | 0.941 ± 0.003 | 0.989 ± 0.001 |
| random, row by row | all of them | 0.939 ± 0.002 | 0.987 ± 0.001 |

- **The split alone explains little:** row by row instead of by capture, macro-F1 moves by +0.017.
- **Source IP alone explains little:** +0.032 on top of the random split.
- **The time and stream columns explain much of the gap:** +0.476, to 0.941. Each capture was recorded at its own time, so absolute time can identify the capture, hence the class; a deployed detector cannot use that. This is why the study drops these columns.
- **A split by capture does not stop this leak** (0.842 with all identifiers). The likely reason: the test rows of 8 of 11 sub-classes are a block of the same capture as their train rows.
- **What this does not show:** that any published score is inflated. Alatram et al. list these columns but do not say which ones the classifiers received; Zhang et al. use the payload, which this measurement does not touch; Parra & Bayas give their classifier four traffic statistics, none of these columns. It covers one classifier and the sampled packets of the study.

## Scores per seed

| Split | Identifiers | Seed | Macro-F1 | Accuracy | Binary F1 | Share of RF importance of the added columns |
|---|---|---|---|---|---|---|
| capture | none | 0 | 0.4472 | 0.4804 | 0.7865 | 0.000 |
| capture | none | 1 | 0.4483 | 0.4811 | 0.7856 | 0.000 |
| capture | none | 2 | 0.4475 | 0.4806 | 0.7864 | 0.000 |
| capture | all | 0 | 0.8434 | 0.8531 | 0.9734 | 0.661 |
| capture | all | 1 | 0.8383 | 0.8479 | 0.9727 | 0.658 |
| capture | all | 2 | 0.8447 | 0.8544 | 0.9748 | 0.663 |
| random | none | 0 | 0.4648 | 0.4946 | 0.7986 | 0.000 |
| random | none | 1 | 0.4632 | 0.4927 | 0.8000 | 0.000 |
| random | none | 2 | 0.4652 | 0.4954 | 0.7993 | 0.000 |
| random | ip | 0 | 0.4964 | 0.5251 | 0.8471 | 0.242 |
| random | ip | 1 | 0.4970 | 0.5248 | 0.8488 | 0.246 |
| random | ip | 2 | 0.4956 | 0.5245 | 0.8476 | 0.239 |
| random | time_stream | 0 | 0.9410 | 0.9411 | 0.9891 | 0.572 |
| random | time_stream | 1 | 0.9366 | 0.9366 | 0.9885 | 0.586 |
| random | time_stream | 2 | 0.9451 | 0.9452 | 0.9906 | 0.593 |
| random | all | 0 | 0.9399 | 0.9399 | 0.9858 | 0.666 |
| random | all | 1 | 0.9364 | 0.9365 | 0.9864 | 0.671 |
| random | all | 2 | 0.9399 | 0.9400 | 0.9873 | 0.664 |

## How each study was read

| Study | Citation | How it was read |
|---|---|---|
| Alatram et al. 2023 | Alatram, Sikos, Johnstone, Szewczyk, Kang. Computer Networks 231:109809, 2023 | full text, published version (CC BY, figshare copy) |
| Zhang et al. 2026 | Zhang, Cui, Cui, Huang, Li, Shi, Lan. Cybersecurity 9:65, 2026 | full text (open access), Tables 4, 5, 6 |
| Parra & Bayas 2026 | Parra, Bayas. Technologies 14(5):253, 2026 | full text (open access), Tables 3, 4 |

## Opened, and evaluated on other data

| Work | Data |
|---|---|
| Swain et al. 2026, Cluster Computing 29:495 | MQTTSET, MQTT-IoT-IDS2020, MQTTEEB-D (abstract); federated |
| Federated hybrid IDS, CCIC 2026 (doi:10.1109/CCIC68129.2026.11486056) | MQTT-IoT-IDS2020 (abstract); federated |
| Solanki & Gupta 2026, Ann. Math. Artif. Intell. 94:395 | MQTT-IoT-IDS2020 (abstract) |
| Drozdev et al. 2026, ICIEAM | MQTTset (abstract) |
| KouhiRonaghi et al. 2026, J. Supercomputing 82:334 | their own BCCC-IoT-MQTT-IDS-Augmented-2025 (abstract) |
| ICITCOM 2026 (doi:10.1109/ICITCOM70739.2026.11714330) | their own testbed (abstract) |
| Putro et al. 2025, IEEE OJCOMS 6:7855 | names this dataset in its related-work discussion only |
| Reddy & Kumar 2026, Frontiers in Big Data 9 | a review: cites this dataset, trains no model on it |
