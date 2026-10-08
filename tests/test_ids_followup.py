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


def test_render_has_both_items():
    r = {"plain": {"macro_f1": [0.4, 0.01], "bin_f1": [0.9, 0.0], "rare_recall": [0.3, 0.0]}, "weighted": {"macro_f1": [0.45, 0.01], "bin_f1": [0.92, 0.0], "rare_recall": [0.35, 0.0]}, "weights": [1, 1]}
    out = {"item5": {"MG-eps5|rf": {"base": r, "rows": r}},
           "item6": {"n": 15, "summary": {"local": {"macro_f1": [0.3, 0.1], "bin_f1": [0.8, 0.1], "rare_recall": [0.2, 0.1]},
                                          "MG-eps5": {"macro_f1": [0.4, 0.1], "bin_f1": [0.85, 0.1], "rare_recall": [0.3, 0.1], "gain_macro_f1": [0.1, 0.05], "clients_better": 14},
                                          "only:MG-eps5": {"macro_f1": [0.41, 0.0], "bin_f1": [0.9, 0.0], "rare_recall": [0.3, 0.0], "gain_macro_f1": [0.11, 0.1], "clients_better": 12}}}}
    md = idf.render(out)
    assert "| MG-eps5 | rf | 0.400 ± 0.010 | 0.450 ± 0.010 |" in md and "| local + MG-eps5 |" in md and "| 14 |" in md
    assert "| MG-eps5 only (same model for every client) | 0.410 ± 0.000 |" in md and "| 12 |" in md
