import numpy as np
import pandas as pd

from brittle.stats import best_of_n, exceeds_floor, item_ci, mean_ci, paired_ci


def frame(per_task):
    return pd.DataFrame([{"task": t, "id": f"{t}-{i}", "v": float(x)}
                         for t, vals in per_task.items() for i, x in enumerate(vals)])


def test_mean_ci_constant_and_seeded():
    assert mean_ci([1, 1, 1]) == {"mean": 1.0, "lo": 1.0, "hi": 1.0, "n": 3}
    x = np.random.default_rng(1).integers(0, 2, 60)
    a = mean_ci(x)
    assert a["lo"] <= a["mean"] <= a["hi"] and mean_ci(x) == a


def test_stratified_bootstrap_keeps_task_mix():
    # all-0 task and all-1 task: a stratified draw always has half of each, so the interval is a point
    df = frame({"a": [0] * 20, "b": [1] * 20})
    r = item_ci(df, "v", n_boot=500)
    assert r["mean"] == r["lo"] == r["hi"] == 0.5


def test_item_rows_are_averaged_before_resampling():
    df = pd.DataFrame([{"task": "a", "id": "x", "v": 1.0}, {"task": "a", "id": "x", "v": 0.0},
                       {"task": "a", "id": "y", "v": 1.0}])
    assert item_ci(df, "v")["n"] == 2 and item_ci(df, "v")["mean"] == 0.75


def test_paired_ci_and_floor_rule():
    a = frame({"a": [1, 1, 0, 1] * 10})
    b = frame({"a": [0, 1, 0, 0] * 10})
    d = paired_ci(a, "v", b)
    assert abs(d["mean"] - 0.5) < 1e-12 and d["lo"] > 0
    assert exceeds_floor(d)
    assert not exceeds_floor(paired_ci(a, "v", a))


def test_best_of_n_equal_variants_shows_selection_gain_only():
    # five variants with identical true accuracy: the winner looks better on the half it was picked on,
    # and the gain disappears on the other half
    rng = np.random.default_rng(3)
    acc = (rng.random((200, 5)) < 0.6).astype(float)
    r = best_of_n(acc, n_splits=400, seed=1)
    assert r["selected"].mean() > 0.03
    assert abs(r["heldout"].mean()) < 0.03
    assert (r["selected"] >= 0).all()


def test_best_of_n_real_improvement_survives():
    acc = np.zeros((100, 3))
    acc[:, 0] = np.arange(100) % 2 == 0      # base 50%
    acc[:, 1] = np.arange(100) % 10 != 0     # variant 1 90%
    acc[:, 2] = np.arange(100) % 3 == 0
    r = best_of_n(acc, n_splits=200)
    assert (r["picks"] == 1).all()
    assert abs(r["heldout"].mean() - 0.4) < 0.05
