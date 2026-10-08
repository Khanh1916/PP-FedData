"""Follow-up round P2: IDS-side class weights and the report."""
import numpy as np

from ppfeddata import ids_followup as idf


def test_weights_chosen_on_validation_fix_a_biased_decision():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], [800, 100, 100])
    P = np.full((len(y), 3), 0.1)
    P[np.arange(len(y)), y] = 0.5
    P[y == 0, 1] = 0.6                                  # class 0 is mistaken for class 1 by a model trained on balanced data
    P += rng.random(P.shape) * 0.01
    classes = ["A", "B", "C"]
    base = idf._metrics(y, P.argmax(1), classes, [1, 2])["macro_f1"]
    w = idf.choose_weights([P], y, classes, [1, 2])
    assert idf._metrics(y, (P * w).argmax(1), classes, [1, 2])["macro_f1"] > base and w[0] / w[1] > 1     # class 0 favoured against class 1


def test_report_section_9_11_and_its_bullet(tmp_path):
    import json

    from ppfeddata import interpret_report as ir
    cfg = {"compute": {"runs_csv": str(tmp_path / "runs.csv")}}
    assert ir._followup_facts(cfg) is None and ir._followup(cfg) == []                             # nothing before the round ran
    row = lambda c, u, f, **kw: {"config": c, "unit": u, "eps": 5.0, "tstr_f1": f, "tstr_f1_std": 0.01, **kw}       # noqa: E731
    pu = {"units": [row("MGs-eps5", "packet", 0.41, m=1), row("MGs-strm2-eps5", "stream", 0.37, m=2), row("MGs-cap25-eps5", "capture", 0.20, m=25)],
          "thresholds": [{**row("MGs-eps5", None, 0.41, eps_one_honest=14.2), "t": 5}, {**row("MGs-t3-eps5", None, 0.405, eps_all_honest=3.7, eps_one_honest=10.1), "t": 3}]}
    m = lambda a, b: {"macro_f1": [a, 0.01], "bin_f1": [0.9, 0.0], "rare_recall": [b, 0.0]}                          # noqa: E731
    i5 = {"MG-eps5|rf": {"base": {"plain": m(0.41, 0.30), "weighted": m(0.43, 0.38), "weights": [1] * 6},
                         "rows": {"plain": m(0.40, 0.30), "weighted": m(0.44, 0.35), "weights": [1] * 6}}}
    i6 = {"n": 15, "summary": {"local": m(0.387, 0.29), "MG-eps5": {**m(0.422, 0.30), "gain_macro_f1": [0.036, 0.04], "clients_better": 15}}}
    (tmp_path / "privacy_units.json").write_text(json.dumps(pu), encoding="utf-8")
    (tmp_path / "ids_followup.json").write_text(json.dumps({"item5": i5, "item6": i6}), encoding="utf-8")
    f = ir._followup_facts(cfg)
    b = ir._followup_bullet(f)
    assert "costs FedDP-Marginal 0.04-0.04" in b and "3 honest clients of 5 changes it by -0.005 to -0.005" in b
    assert "MG-eps5 RF 0.410 → 0.430" in b and "+0.036 macro-F1" in b and "15 of 15 client models" in b
    md = "\n".join(ir._followup(cfg))
    assert "### 9.11" in md and "| 5 | 0.410 ± 0.010 | 0.370 ± 0.010 (m 2) | 0.200 ± 0.010 (m 25) |" in md
    assert "| 5 | 3 | 0.405 ± 0.010 | 3.70 | 10.1 |" in md and "| 5 | 5 (as before) | 0.410 ± 0.010 | 5.00 | 14.2 |" in md
    assert "| MG-eps5 | RF | 0.410 | 0.430 | 0.440 (all rows + weights) | 0.300 → 0.350 |" in md and "| local + MG-eps5 | 0.422 ± 0.010 | 0.300 | +0.036 | 15 / 15 |" in md


def test_render_has_both_items():
    r = {"plain": {"macro_f1": [0.4, 0.01], "bin_f1": [0.9, 0.0], "rare_recall": [0.3, 0.0]}, "weighted": {"macro_f1": [0.45, 0.01], "bin_f1": [0.92, 0.0], "rare_recall": [0.35, 0.0]}, "weights": [1, 1]}
    out = {"item5": {"MG-eps5|rf": {"base": r, "rows": r}},
           "item6": {"n": 15, "summary": {"local": {"macro_f1": [0.3, 0.1], "bin_f1": [0.8, 0.1], "rare_recall": [0.2, 0.1]},
                                          "MG-eps5": {"macro_f1": [0.4, 0.1], "bin_f1": [0.85, 0.1], "rare_recall": [0.3, 0.1], "gain_macro_f1": [0.1, 0.05], "clients_better": 14},
                                          "only:MG-eps5": {"macro_f1": [0.41, 0.0], "bin_f1": [0.9, 0.0], "rare_recall": [0.3, 0.0], "gain_macro_f1": [0.11, 0.1], "clients_better": 12}}}}
    md = idf.render(out)
    assert "| MG-eps5 | rf | 0.400 ± 0.010 | 0.450 ± 0.010 |" in md and "| local + MG-eps5 |" in md and "| 14 |" in md
    assert "| MG-eps5 only (same model for every client) | 0.410 ± 0.000 |" in md and "| 12 |" in md


def test_section_9_11_names_the_threshold_picked_by_recommend(tmp_path):
    import json

    from ppfeddata import interpret_report as ir
    cfg = {"compute": {"runs_csv": str(tmp_path / "runs.csv")}}
    row = lambda c, t, f: {"config": c, "eps": 5.0, "tstr_f1": f, "tstr_f1_std": 0.01, "t": t}      # noqa: E731
    (tmp_path / "privacy_units.json").write_text(json.dumps({"thresholds": [row("MGs-eps5", 5, 0.41), row("MGs-t3-eps5", 3, 0.41)]}), encoding="utf-8")
    rec = [{"requirement": {"name": "gateway-dropout", "max_eps": 5.0, "honest_clients": 3}, "best": {"label": "MGs-t3-eps5", "tstr_f1": 0.41, "eps_eff": 5.0}},
           {"requirement": {"name": "lab", "max_eps": None}, "best": {"label": "B3"}}]
    (tmp_path / "recommend.json").write_text(json.dumps(rec), encoding="utf-8")
    md = "\n".join(ir._followup(cfg))
    assert "The default stays t = K = 5" in md and "`gateway-dropout` (ε ≤ 5 with 3 clients adding their share) → MGs-t3-eps5 (TSTR macro-F1 0.410, ε 5.00)" in md
    assert "`lab`" not in md
