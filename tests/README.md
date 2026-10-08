# `tests/` — pytest for every stage

```bash
python -m pytest -q
```

The whole suite takes about 10 minutes on a laptop CPU and needs no raw data: the tests build small synthetic datasets with the real
column layout, and a few read the committed results (`results/*.json`) to check that the reports and READMEs match them. Some tests start
real Flower simulations with Ray; on Windows a Ray worker can occasionally hang (the progress stops and the CPU stays idle): stop the run
and start it again. Run one file with `python -m pytest -q tests/test_marginal.py`.

| File | Covers |
|---|---|
| `test_config.py`, `test_seed.py` | config loading and hashing, deterministic seeding |
| `test_inventory.py`, `test_harmonize.py` | Phases 1-2: inventory, schema harmonisation, format audit |
| `test_split_sample.py`, `test_preprocess.py` | Phases 3-4: group split, streaming sampling, multi-value parsing, preprocessing |
| `test_leakage.py` | Phase 5: leakage checks on data with leakage planted on purpose |
| `test_eval.py` | Phase 6: metrics, protocols, bootstrap, fidelity, privacy, overhead, run ledger, Markdown tables |
| `test_models.py` | Phase 7: CVAE shapes, loss, generation, conditioning, reproducibility |
| `test_fl.py` | Phase 8: Dirichlet partition, FedAvg, checkpoint / resume through a real Flower simulation |
| `test_dp.py` | Phase 9: DP-SGD accounting and training, DP resume through Flower |
| `test_secagg.py`, `test_secagg_compact.py` | Phase 10: SecAgg+ correctness, hiding, dropout, compact uint16 / uint32 encodings |
| `test_experiment.py` | Phase 11: the experiment matrix, planning against the ledger, resume |
| `test_aggregate.py`, `test_interpret.py` | Phases 11-12: summary, figures, final report, interpretation rules R1-R6 and red flags |
| `test_demo.py`, `test_accept.py` | Phase 13: limitations, demo logic, generated README blocks and links, report links, acceptance checks |
| `test_fed_classifier.py`, `test_mia.py` | follow-ups: the IDS trained by FL, sensitivity worlds, membership attacks with model access |
| `test_scorecard.py`, `test_o1.py`, `test_taug_rare.py` | optimisation O0-O1: scorecard and Pareto front, DP residual statistics, TAugR |
| `test_dpfedsgd.py` | optimisation O2: DP-FedSGD with the noise split over the clients |
| `test_marginal.py`, `test_marginal_bn.py` | optimisation O3: FedDP-Marginal (accounting, tree, sampling) and its Bayesian-network option |
| `test_recommend.py` | optimisation O4: configurations per deployment requirement |
| `test_group_dp.py`, `test_ids_followup.py` | follow-up round P1-P2: group-level DP, honest-client threshold, IDS-side options, local augmentation, report section 9.11 |
| `test_docs.py` | the folder READMEs and the architecture documents name only modules, files and commands that exist |
