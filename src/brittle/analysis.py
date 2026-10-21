"""Tables of flips, accuracy changes and best-prompt selection, computed from results/answers/."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from . import config as C
from .data import load_items
from .parse import check
from .prompts import KIND, KINDS, render, to_original, variants
from .stats import best_of_n, item_ci, paired_ci

NONE = "<no answer>"


def load_rows(split: str = "main") -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(C.RESULTS, "answers", "*.jsonl"))):
        is_pilot = p.endswith("_pilot.jsonl")
        if (split == "pilot") != is_pilot:
            continue
        with open(p) as f:
            rows += [json.loads(line) for line in f]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # re-parse every stored output with the current parser, so all rows are scored by the same rules
    items = {it["id"]: it for it in load_items(None)}
    parsed = []
    for r in df.itertuples():
        it = items[r.id]
        rd = render(it, r.variant)
        chk = check(r.task, r.output, rd.gold, rd.options)
        ans = to_original(it, chk["answer"], rd.perm) if r.task == "mmlu" else chk["answer"]
        parsed.append((ans, bool(chk["correct"]), chk["method"]))
    df["answer"], df["correct"], df["method"] = zip(*parsed)
    df["answer"] = df["answer"].fillna(NONE)
    df["correct"] = df["correct"].astype(float)
    return with_base(complete_items(df))


def complete_items(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a model's item only if every variant of it was answered, so all comparisons are paired."""
    need = df.task.map({t: len(variants(t)) + 1 for t in C.TASKS})
    n = df.groupby(["model", "id"]).variant.transform("nunique")
    return df[n == need].copy()


def with_base(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.variant == "base"][["model", "id", "answer", "correct"]].rename(
        columns={"answer": "base_answer", "correct": "base_correct"})
    df = df.merge(base, on=["model", "id"], how="inner")
    df["flip"] = (df.answer != df.base_answer).astype(float)
    df["r2w"] = ((df.base_correct == 1) & (df.correct == 0)).astype(float)
    df["w2r"] = ((df.base_correct == 0) & (df.correct == 1)).astype(float)
    df["dacc"] = df.correct - df.base_correct
    return df


def models_in(df: pd.DataFrame) -> list[str]:
    return [m for m in C.MODELS if m in set(df.model)]


def common_items(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Rows of `models` restricted to the items every one of them completed."""
    ids = None
    for m in models:
        got = set(df[df.model == m].id)
        ids = got if ids is None else ids & got
    return df[df.model.isin(models) & df.id.isin(ids or set())]


def main_models(df: pd.DataFrame) -> list[str]:
    """Models run without thinking, which share the main grid."""
    return [m for m in models_in(df) if m not in C.THINKING_ONLY]


def rephrased(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.variant.isin(["base", "aa"])]


def aa_floor(df: pd.DataFrame, model: str) -> dict:
    d = df[(df.model == model) & (df.variant == "aa")]
    return {**item_ci(d, "flip"), "flips": int(d.flip.sum()), "acc_diff": float(d.dacc.mean())}


def kind_table(df: pd.DataFrame, model: str | None, value: str = "flip", tasks: list[str] | None = None) -> list[dict]:
    """Per kind: mean over items of `value` (an item's variants of that kind averaged), with its excess
    over the A/A repeat, paired by item."""
    d = df if model is None else df[df.model == model]
    if tasks:
        d = d[d.task.isin(tasks)]
    aa = d[d.variant == "aa"]
    out = []
    for k in KINDS:
        dk = d[d.kind == k]
        if dk.empty:
            continue
        ref = aa[aa.id.isin(set(dk.id))]
        est = item_ci(dk, value)
        exc = paired_ci(dk, value, ref)
        out.append({"kind": k, **est, "excess": exc, "tasks": sorted(set(dk.task))})
    return out


def variant_table(df: pd.DataFrame, model: str, task: str) -> pd.DataFrame:
    d = df[(df.model == model) & (df.task == task)]
    rows = []
    for v in variants(task) + ["aa"]:
        dv = d[d.variant == v]
        if dv.empty:
            continue
        acc = item_ci(dv, "correct")
        rows.append({"variant": v, "kind": KIND[v], "acc": acc["mean"], "acc_lo": acc["lo"], "acc_hi": acc["hi"],
                     "flip": dv.flip.mean(), "r2w": dv.r2w.sum(), "w2r": dv.w2r.sum(),
                     "dacc": paired_ci(dv, "correct", d[d.variant == "base"]),
                     "noparse": float((dv.answer == NONE).mean()), "n": len(dv)})
    return pd.DataFrame(rows)


def item_stability(df: pd.DataFrame, model: str) -> dict:
    """Per item, over the base prompt and its rephrasings (not the A/A repeat)."""
    d = df[(df.model == model) & (df.variant != "aa")]
    g = d.groupby(["task", "id"])
    share = g.correct.mean()
    n_answers = g.answer.nunique()
    return {
        "always_right": float((share == 1).mean()), "always_wrong": float((share == 0).mean()),
        "mixed": float(((share > 0) & (share < 1)).mean()), "n": int(len(share)),
        "any_flip": float((n_answers > 1).mean()),
        "spread_sd": float(share[(share > 0) & (share < 1)].std()) if ((share > 0) & (share < 1)).any() else 0.0,
        "by_task": {t: {"always_right": float((share.loc[t] == 1).mean()), "always_wrong": float((share.loc[t] == 0).mean()),
                        "mixed": float(((share.loc[t] > 0) & (share.loc[t] < 1)).mean()),
                        "any_flip": float((n_answers.loc[t] > 1).mean())}
                    for t in sorted(set(d.task))},
    }


def acc_matrix(df: pd.DataFrame, model: str, task: str | None, vs: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    d = df[(df.model == model)]
    if task:
        d = d[d.task == task]
    piv = d[d.variant.isin(vs)].pivot_table(index=["task", "id"], columns="variant", values="correct")
    piv = piv[[v for v in vs if v in piv.columns]].dropna()
    tasks = piv.index.get_level_values(0).values
    return piv.values, list(piv.columns), tasks


def best_prompt(df: pd.DataFrame, model: str, task: str | None) -> dict:
    if task is None and set(df[df.model == model].task) != set(C.TASKS):
        return {}
    vs = variants(task) if task else [v for v in variants("gsm8k") if all(v in variants(t) for t in C.TASKS)]
    acc, cols, tasks = acc_matrix(df, model, task, vs)
    if acc.size == 0:
        return {}
    if task is None:
        # stratified halves: shuffle within each task so each half holds half of every task
        order = np.argsort(tasks, kind="stable")
        acc, tasks = acc[order], tasks[order]
        r = _best_of_n_stratified(acc, tasks, cols.index("base"))
    else:
        r = best_of_n(acc, base=cols.index("base"))
    picks = pd.Series([cols[i] for i in r["picks"]]).value_counts(normalize=True)
    full = acc.mean(axis=0)
    w = int(np.argmax(full))
    return {
        "n_items": int(acc.shape[0]), "n_variants": len(cols),
        "selected": float(r["selected"].mean()), "selected_lo": float(np.percentile(r["selected"], 2.5)),
        "selected_hi": float(np.percentile(r["selected"], 97.5)),
        "heldout": float(r["heldout"].mean()), "heldout_lo": float(np.percentile(r["heldout"], 2.5)),
        "heldout_hi": float(np.percentile(r["heldout"], 97.5)),
        "p_heldout_pos": float((r["heldout"] > 0).mean()), "p_heldout_neg": float((r["heldout"] < 0).mean()),
        "p_base_picked": float((np.array(r["picks"]) == cols.index("base")).mean()),
        "top_pick": picks.index[0], "top_pick_share": float(picks.iloc[0]),
        "full_best": cols[w], "full_best_gain": float(full[w] - full[cols.index("base")]),
        "base_acc": float(full[cols.index("base")]), "full_spread": float(full.max() - full.min()),
    }


def _best_of_n_stratified(acc: np.ndarray, tasks: np.ndarray, base: int, n_splits: int = C.SPLITS,
                          seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(tasks == t) for t in np.unique(tasks)]
    sel, held, picks = np.zeros(n_splits), np.zeros(n_splits), np.zeros(n_splits, dtype=int)
    for s in range(n_splits):
        a, b = [], []
        for g in groups:
            p = rng.permutation(g)
            a.append(p[:len(g) // 2])
            b.append(p[len(g) // 2:])
        a, b = np.concatenate(a), np.concatenate(b)
        sa, sb = acc[a].mean(axis=0), acc[b].mean(axis=0)
        w = int(rng.choice(np.flatnonzero(sa == sa.max())))
        sel[s], held[s], picks[s] = sa[w] - sa[base], sb[w] - sb[base], w
    return {"selected": sel, "heldout": held, "picks": picks}


LENIENT_DROP = {"the", "food", "restaurant", "pub", "coffee", "shop", "cuisine"}


def lenient(answer: str) -> str:
    """Extraction answer with generic words dropped, to see how many flips are only span boundaries
    ("English food" against "English")."""
    if answer == NONE:
        return answer
    d = json.loads(answer)
    return json.dumps({k: " ".join(w for w in v.split() if w not in LENIENT_DROP) for k, v in d.items()},
                      sort_keys=True)


def extraction_boundary_share(df: pd.DataFrame) -> dict:
    d = rephrased(df)
    d = d[(d.task == "extract") & (d.flip == 1)]
    if d.empty:
        return {"flips": 0, "boundary": 0}
    same = [lenient(a) == lenient(b) for a, b in zip(d.answer, d.base_answer)]
    return {"flips": int(len(d)), "boundary": int(sum(same))}


def meaning_check(df: pd.DataFrame, models: list[str]) -> list[dict]:
    """A rephrasing that changed what is asked should cost accuracy for every model. List each variant's
    accuracy change per model, and flag variants whose interval is below zero for all of them."""
    out = []
    for task in C.TASKS:
        if df[df.task == task].empty:
            continue
        for v in variants(task):
            if v == "base":
                continue
            per = {}
            for m in models:
                d = df[(df.model == m) & (df.task == task)]
                per[m] = paired_ci(d[d.variant == v], "correct", d[d.variant == "base"])
            all_down = all(x["hi"] < 0 for x in per.values()) if per else False
            out.append({"task": task, "variant": v, "per_model": per, "all_down": all_down})
    return out
