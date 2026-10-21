"""Bootstrap statistics that resample items, never single calls.

Adapted from memory-rot and agent-interface-lab (bootstrap by unit, A/A gap). Here the unit is the
item: every rephrasing of an item shares its question, so an item's rows move together. When tasks
are pooled, items are resampled within each task so every draw keeps 100 items per task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def boot_means(values: np.ndarray, groups: np.ndarray | None = None, n_boot: int = C.BOOT,
               seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the mean of per-item values, stratified by `groups`."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.zeros(len(values), dtype=int)
    groups = np.asarray(groups)
    sums = np.zeros(n_boot)
    for g in np.unique(groups):
        v = values[groups == g]
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        sums += v[idx].sum(axis=1)
    return sums / len(values)


def mean_ci(values, groups=None, n_boot: int = C.BOOT, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    b = boot_means(values, groups, n_boot, seed)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(values))}


def item_ci(df: pd.DataFrame, value: str, n_boot: int = C.BOOT, seed: int = 0) -> dict:
    """Mean of `value` over items (averaging an item's rows first), item bootstrap stratified by task."""
    per = df.groupby(["task", "id"])[value].mean().reset_index()
    return mean_ci(per[value].values, per["task"].values, n_boot, seed)


def paired_ci(df: pd.DataFrame, value: str, ref: pd.DataFrame, n_boot: int = C.BOOT, seed: int = 0) -> dict:
    """Mean over items of (item mean of `value` in df) minus (item mean in ref), paired by item."""
    a = df.groupby(["task", "id"])[value].mean()
    b = ref.groupby(["task", "id"])[value].mean()
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").reset_index()
    d = (m["a"] - m["b"]).values
    return mean_ci(d, m["task"].values, n_boot, seed)


def exceeds_floor(diff: dict) -> bool:
    """A rephrasing counts only if its excess over the A/A floor has an interval above zero."""
    return bool(diff and diff["n"] and diff["lo"] > 0)


def best_of_n(acc: np.ndarray, base: int = 0, n_splits: int = C.SPLITS, seed: int = 0) -> dict:
    """Winner's curse for prompts.

    `acc` is an items x variants 0/1 matrix. For each random half split, the variant with the highest
    accuracy on the selection half is picked (ties broken at random) and its gain over the base prompt
    is measured on both halves. Returns the per-split selected and held-out gains.
    """
    acc = np.asarray(acc, dtype=float)
    n, k = acc.shape
    rng = np.random.default_rng(seed)
    sel, held, picks = np.zeros(n_splits), np.zeros(n_splits), np.zeros(n_splits, dtype=int)
    half = n // 2
    for s in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[:half], perm[half:]
        sa, sb = acc[a].mean(axis=0), acc[b].mean(axis=0)
        best = np.flatnonzero(sa == sa.max())
        w = int(rng.choice(best))
        sel[s], held[s], picks[s] = sa[w] - sa[base], sb[w] - sb[base], w
    return {"selected": sel, "heldout": held, "picks": picks}
