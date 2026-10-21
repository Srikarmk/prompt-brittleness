"""reports/results.md and reports/figures/, generated from results/ and the item file only."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import analysis as A  # noqa: E402
from . import config as C  # noqa: E402
from .data import load_items  # noqa: E402
from .prompts import KINDS, render, variants  # noqa: E402
from .stats import item_ci  # noqa: E402

MODEL_COLOR = {"qwen3:8b": "#2a78d6", "llama3.1:8b": "#eb6834", "gemma3:12b": "#1baf7a", "qwen3:30b-a3b": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
R2W, W2R = "#eb6834", "#2a78d6"
TASK_LABEL = {"gsm8k": "GSM8K", "mmlu": "MMLU", "extract": "E2E extraction"}


def mlabel(m: str) -> str:
    return f"{m} (thinking)" if m in C.THINKING_ONLY else m


def pct(x, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def ci(r: dict, d: int = 1) -> str:
    if not r.get("n", 1):
        return ""
    return f"{pct(r['mean'], d)} [{100 * r['lo']:.{d}f}, {100 * r['hi']:.{d}f}]"


def pp(r: dict, d: int = 1) -> str:
    return f"{100 * r['mean']:+.{d}f} pp [{100 * r['lo']:+.{d}f}, {100 * r['hi']:+.{d}f}]"


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]


def style_axes(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(C.FIGURES, exist_ok=True)
    fig.savefig(os.path.join(C.FIGURES, name), dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figures

def fig_flip_by_kind(kt: dict, floors: dict, models: list[str]):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    n = len(models)
    for j, m in enumerate(models):
        rows = {r["kind"]: r for r in kt[m]}
        ys, xs, lo, hi = [], [], [], []
        for i, k in enumerate(KINDS):
            if k in rows:
                ys.append(i + (j - (n - 1) / 2) * 0.17)
                xs.append(100 * rows[k]["mean"])
                lo.append(100 * (rows[k]["mean"] - rows[k]["lo"]))
                hi.append(100 * (rows[k]["hi"] - rows[k]["mean"]))
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", color=MODEL_COLOR[m], markersize=6, elinewidth=1.4,
                    capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.2, label=mlabel(m))
        f = 100 * floors[m]["mean"]
        ax.axvline(f, color=MODEL_COLOR[m], linewidth=1, linestyle=(0, (2, 2)), alpha=0.9)
    ax.set_yticks(range(len(KINDS)), KINDS)
    ax.invert_yaxis()
    ax.set_xlabel("answers that changed vs the base prompt (% of items, 95% CI)", fontsize=8, color=INK2)
    ax.set_title("Flip rate by kind of rephrasing (dashed: same prompt sent twice)", fontsize=10, color=INK,
                 loc="left")
    ax.set_xlim(left=0)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(models))
    save(fig, "flip_by_kind.png")


def fig_r2w(r2w: dict, w2r: dict, models: list[str]):
    fig, axes = plt.subplots(1, len(models), figsize=(2.9 * len(models) + 1.2, 4.0), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    lim = max(4, 1.15 * max(100 * r["mean"] for m in models for r in r2w[m] + w2r[m]))
    for ax, m in zip(axes[0], models):
        style_axes(ax)
        a = {r["kind"]: r for r in r2w[m]}
        b = {r["kind"]: r for r in w2r[m]}
        ks = [k for k in KINDS if k in a]
        y = np.arange(len(ks))
        ax.barh(y, [-100 * a[k]["mean"] for k in ks], color=R2W, height=0.62, label="right to wrong")
        ax.barh(y, [100 * b[k]["mean"] for k in ks], color=W2R, height=0.62, label="wrong to right")
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(y, ks)
        ax.set_title(mlabel(m), fontsize=9, color=INK)
        ax.set_xlabel("% of items", fontsize=8, color=INK2)
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    axes[0][0].invert_yaxis()
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Which way the answers moved: right to wrong (left) vs wrong to right (right)", fontsize=10,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "right_to_wrong.png")


def fig_best_prompt(bp: dict, models: list[str]):
    cells = [(m, t) for m in models for t in C.TASKS + [None] if bp.get((m, t))]
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(cells) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for i, (m, t) in enumerate(cells):
        r = bp[(m, t)]
        ax.plot([100 * r["selected"], 100 * r["heldout"]], [i, i], color=GRID, linewidth=2, zorder=1)
        ax.errorbar(100 * r["heldout"], i, xerr=[[100 * (r["heldout"] - r["heldout_lo"])],
                                                [100 * (r["heldout_hi"] - r["heldout"])]],
                    fmt="o", color=MODEL_COLOR[m], markersize=6, elinewidth=1.2, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.plot(100 * r["selected"], i, marker="o", markersize=6, markerfacecolor=SURFACE,
                markeredgecolor=MODEL_COLOR[m], markeredgewidth=1.6, linestyle="none", zorder=3)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(cells)), [f"{mlabel(m)}, {TASK_LABEL.get(t, 'all three tasks')}" for m, t in cells])
    ax.invert_yaxis()
    ax.set_xlabel("gain of the best-looking prompt over the base prompt (pp)", fontsize=8, color=INK2)
    ax.set_title("Best of N prompts: open = gain on the half it was picked on, filled = gain on the other half",
                 fontsize=9.5, color=INK, loc="left")
    save(fig, "best_prompt_selection.png")


def fig_accuracy_spread(vt: dict, models: list[str]):
    fig, axes = plt.subplots(1, len(C.TASKS), figsize=(9.6, 3.4), sharey=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, t in zip(axes, C.TASKS):
        style_axes(ax, grid_axis="y")
        for j, m in enumerate(models):
            d = vt[(m, t)]
            if d.empty:
                continue
            d = d[d.variant != "aa"]
            x = np.full(len(d), j) + np.linspace(-0.18, 0.18, len(d))
            ax.scatter(x, 100 * d.acc, s=16, color=MODEL_COLOR[m], alpha=0.75, edgecolors="none")
            b = d[d.variant == "base"].acc.iloc[0]
            ax.plot([j - 0.28, j + 0.28], [100 * b, 100 * b], color=INK, linewidth=1.6)
        ax.set_xticks(range(len(models)), [mlabel(m).replace(" (thinking)", "\n(thinking)") for m in models],
                      fontsize=7)
        ax.set_title(TASK_LABEL[t], fontsize=9, color=INK)
        ax.set_ylabel("accuracy (%)", fontsize=8, color=INK2)
    fig.suptitle("Accuracy under each prompt variant (dots); black line = base prompt", fontsize=10, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "accuracy_by_variant.png")


# ---------------------------------------------------------------- report

def write_report() -> None:
    df = A.load_rows("main")
    if df.empty:
        raise SystemExit("no results yet")
    models = A.models_in(df)
    full = A.main_models(df)
    common = A.common_items(df, full)
    items = load_items("main")
    n_items = {t: int(df[df.task == t].id.nunique()) for t in C.TASKS}
    floors = {m: A.aa_floor(df, m) for m in models}
    kt = {m: A.kind_table(df, m) for m in models}
    r2w = {m: A.kind_table(df, m, "r2w") for m in models}
    w2r = {m: A.kind_table(df, m, "w2r") for m in models}
    dacc = {m: A.kind_table(df, m, "dacc") for m in models}
    vt = {(m, t): A.variant_table(df, m, t) for m in models for t in C.TASKS}
    stab = {m: A.item_stability(df, m) for m in models}
    bp = {(m, t): A.best_prompt(df, m, t) for m in full for t in C.TASKS + [None]}
    reph = A.rephrased(df)
    n_think = int(df[df.model.isin(C.THINKING_ONLY)].id.nunique())

    fig_flip_by_kind(kt, floors, models)
    fig_r2w(r2w, w2r, models)
    fig_best_prompt(bp, full)
    fig_accuracy_spread(vt, models)

    L = ["# Results", "",
         "Generated by `uv run brittle report` from `results/answers/*.jsonl` and `data/items.jsonl`. Every "
         "number below is computed by code from cached model outputs. Intervals are 95% percentile bootstraps "
         f"({C.BOOT:,} resamples) over items, stratified by task when tasks are pooled; `pp` = percentage points. "
         "Every stored output is re-parsed with the current parser when this report is built.", ""]
    meta_path = os.path.join(C.RESULTS, "models.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    # headline
    L += ["## Headline", ""]
    for m in models:
        d = reph[reph.model == m]
        fl = item_ci(d, "flip")
        rw = item_ci(d, "r2w")
        wr = item_ci(d, "w2r")
        s = stab[m]
        L.append(f"- **{mlabel(m)}** ({d.id.nunique()} items): averaged over all rephrasings, the answer changed "
                 f"on {ci(fl)} of items "
                 f"(same prompt sent twice: {ci(floors[m])}). Right to wrong {ci(rw)}, wrong to right {ci(wr)}. "
                 f"{pct(s['any_flip'], 0)} of items gave more than one distinct answer across the prompts, and "
                 f"{pct(s['mixed'], 0)} were right under some prompts and wrong under others.")
    L.append("")
    for m in models:
        top = sorted(kt[m], key=lambda r: -r["mean"])
        real = [r["kind"] for r in top if r["excess"]["lo"] > 0]
        L.append(f"- **{mlabel(m)}**, kinds whose flip rate is above the A/A repeat (interval of the excess above "
                 f"zero): {', '.join(real) if real else 'none'}. Largest: "
                 + "; ".join(f"{r['kind']} {ci(r)}" for r in top[:3]) + ".")
    L.append("")
    for m in models:
        r = bp.get((m, None))
        if r:
            L.append(f"- **{mlabel(m)}**, best of {r['n_variants']} prompt wordings for all three tasks together: "
                     f"picked on half the items it looks {100 * r['selected']:+.1f} pp better than the base prompt; "
                     f"on the other half it is {100 * r['heldout']:+.1f} pp (middle 95% of {C.SPLITS:,} random "
                     f"splits: {100 * r['heldout_lo']:+.1f} to {100 * r['heldout_hi']:+.1f}).")
    L.append("")

    # 1 setup
    L += ["## 1. Setup", ""]
    rows = []
    for m in models:
        d = df[df.model == m]
        info = meta.get(m, {})
        rows.append([mlabel(m), info.get("digest", "")[:12], str(info.get("think", "")), len(d),
                     " / ".join(str(len(d[d.task == t])) for t in C.TASKS),
                     " / ".join(f"{d[d.task == t].seconds.median():.1f}" if (d.task == t).any() else "-"
                                for t in C.TASKS), f"{d.output_tokens.mean():.0f}",
                     pct(d.truncated.mean())])
    L += md_table(["model", "digest", "think", "calls", "calls per task (GSM8K / MMLU / E2E)",
                   "median server seconds per call",
                   "mean output tokens", "hit the length cap"], rows)
    L.append(f"Items analysed (complete under every variant): {n_items['gsm8k']} GSM8K, {n_items['mmlu']} MMLU "
             f"({', '.join(s.replace('_', ' ') for s in C.MMLU_SUBJECTS)}), "
             f"{n_items['extract']} E2E extraction, out of {len(items)} sampled. Prompt variants per item: base, A/A repeat, and "
             f"{len(variants('gsm8k')) - 1} rephrasings ({len(variants('mmlu')) - 1} for MMLU, which adds two "
             "option orders). Items completed by every main model ("""
             + ", ".join(mlabel(m) for m in full) + "): "
             + ", ".join(f"{common[common.task == t].id.nunique()} {TASK_LABEL[t]}" for t in C.TASKS) + ".")
    L.append("")
    rows = []
    for m in models:
        for t in C.TASKS:
            d = df[(df.model == m) & (df.task == t) & (df.variant == "base")]
            if d.empty:
                continue
            meth = d.method.value_counts(normalize=True)
            rows.append([mlabel(m), TASK_LABEL[t], len(d), ci(item_ci(d, "correct")),
                         ", ".join(f"{k} {100 * v:.0f}%" for k, v in meth.items())])
    L += ["Base-prompt accuracy and which parsing rule produced the answer:", ""]
    L += md_table(["model", "task", "items", "accuracy", "parse rule"], rows)

    # 2 floor
    L += ["## 2. Noise floor: the same prompt sent twice", "",
          "The base prompt is sent a second time with a separate cache entry (temperature 0, same seed), spread "
          "through the run so it shares the server with different neighbours. A rephrasing only counts as having "
          "an effect when its flip rate exceeds this repeat with an interval above zero (paired by item).", ""]
    L += md_table(["model", "items", "answers changed", "flip rate", "accuracy change"],
                  [[mlabel(m), floors[m]["n"], floors[m]["flips"], ci(floors[m]), f"{100 * floors[m]['acc_diff']:+.1f} pp"]
                   for m in models])

    # 3 kinds
    L += ["## 3. Which kinds of rephrasing matter", "",
          "Flip rate = share of items whose parsed answer differs from the base prompt's answer (no answer counts "
          "as an answer). For kinds with two variants, an item's value is the mean over both. `excess` = flip rate "
          "minus the A/A repeat on the same items. `*` marks kinds whose excess interval is above zero. Option "
          "order applies to MMLU only, so its rows cover the MMLU items only.", "",
          "![flip rate by kind](figures/flip_by_kind.png)", ""]
    for m in models:
        L += [f"**{mlabel(m)}**", ""]
        rows = []
        for r, a, b, c in zip(kt[m], r2w[m], w2r[m], dacc[m]):
            rows.append([r["kind"] + (" *" if r["excess"]["lo"] > 0 else ""), r["n"], ci(r), pp(r["excess"]),
                         ci(a), ci(b), pp(c)])
        L += md_table(["kind", "items", "flip rate", "excess over A/A", "right to wrong", "wrong to right",
                       "accuracy change"], rows)
    eb = A.extraction_boundary_share(df)
    if eb["flips"]:
        L += [f"Of the {eb['flips']} E2E answers (all models, all rephrasings) that changed, {eb['boundary']} "
              f"({100 * eb['boundary'] / eb['flips']:.0f}%) differ from the base answer only by generic words such "
              "as \"food\", \"the\" or \"restaurant\" (\"English food\" against \"English\"): the span the "
              "model copied moved, not the fact. They still count as flips, and as wrong when the gold is the "
              "shorter span, because the field rule is exact match.", ""]
    L += ["By task (flip rate, the main models averaged per item, on the items all of them completed):", ""]
    pooled = common
    rows = []
    for k in KINDS:
        row = [k]
        for t in C.TASKS:
            d = pooled[(pooled.kind == k) & (pooled.task == t)]
            row.append(ci(item_ci(d, "flip")) if len(d) else "")
        rows.append(row)
    aa_row = ["identity (A/A)"] + [ci(item_ci(pooled[(pooled.variant == "aa") & (pooled.task == t)], "flip"))
                                   for t in C.TASKS]
    L += md_table(["kind"] + [TASK_LABEL[t] for t in C.TASKS], rows + [aa_row])

    # 4 direction
    L += ["## 4. Right to wrong vs wrong to right", "",
          "Shares of all items. Conditional rates: of the items the base prompt got right, how many a rephrasing "
          "got wrong, and of those it got wrong, how many a rephrasing fixed.", "",
          "![right to wrong](figures/right_to_wrong.png)", ""]
    rows = []
    for m in models:
        d = reph[reph.model == m]
        right = d[d.base_correct == 1]
        wrong = d[d.base_correct == 0]
        rows.append([mlabel(m), ci(item_ci(d, "r2w")), ci(item_ci(d, "w2r")), pp(item_ci(d, "dacc")),
                     ci(item_ci(right.assign(x=1 - right.correct), "x")) if len(right) else "n/a",
                     ci(item_ci(wrong.assign(x=wrong.correct), "x")) if len(wrong) else "n/a"])
    L += md_table(["model", "right to wrong", "wrong to right", "net accuracy change",
                   "P(wrong | base right)", "P(right | base wrong)"], rows)

    # 5 spread
    L += ["## 5. Accuracy spread", "",
          "Per item, over the base prompt and its rephrasings (A/A repeat excluded).", "",
          "![accuracy by variant](figures/accuracy_by_variant.png)", ""]
    rows = []
    for m in models:
        s = stab[m]
        for t, b in s["by_task"].items():
            v = vt[(m, t)]
            v = v[v.variant != "aa"]
            rows.append([mlabel(m), TASK_LABEL[t], pct(b["always_right"], 0), pct(b["always_wrong"], 0),
                         pct(b["mixed"], 0), pct(b["any_flip"], 0), pct(v.acc.min(), 0), pct(v.acc.max(), 0),
                         f"{100 * (v.acc.max() - v.acc.min()):.0f} pp"])
    L += md_table(["model", "task", "always right", "always wrong", "right under some prompts only",
                   "more than one answer", "worst prompt", "best prompt", "range"], rows)
    for m in models:
        L += [f"<details><summary>Accuracy per variant, {mlabel(m)}</summary>", ""]
        rows = []
        for t in C.TASKS:
            if vt[(m, t)].empty:
                continue
            for r in vt[(m, t)].itertuples():
                rows.append([TASK_LABEL[t], r.variant, r.kind, f"{pct(r.acc)} [{100 * r.acc_lo:.0f}, {100 * r.acc_hi:.0f}]",
                             pp(r.dacc) if r.variant != "base" else "", pct(r.flip), int(r.r2w), int(r.w2r),
                             pct(r.noparse)])
        L += md_table(["task", "variant", "kind", "accuracy", "vs base", "flip rate", "right to wrong (items)",
                       "wrong to right (items)", "no answer parsed"], rows)
        L += ["</details>", ""]

    # 6 best prompt
    L += ["## 6. The best-scoring prompt: real improvement or lucky draw?", "",
          f"For each model and task, the items are split at random into two halves ({C.SPLITS:,} splits). The prompt "
          "variant with the highest accuracy on the first half is picked (ties broken at random), as a prompt "
          "engineer comparing wordings on a dev set would. Its gain over the base prompt is measured on the half it "
          "was picked on (`selected`) and on the other half (`held out`). The ranges are the middle 95% of splits, "
          "not confidence intervals. `all three tasks` picks one wording (the 11 shared by all tasks) for the "
          "pooled items, halving within each task."
          + (f" The thinking model is left out: its {n_think} items are too few to split." if n_think else ""), "",
          "![best prompt](figures/best_prompt_selection.png)", ""]
    rows = []
    for (m, t), r in bp.items():
        if not r:
            continue
        rows.append([mlabel(m), TASK_LABEL.get(t, "all three tasks"), r["n_items"], r["n_variants"], pct(r["base_acc"]),
                     f"{100 * r['selected']:+.1f} [{100 * r['selected_lo']:+.1f}, {100 * r['selected_hi']:+.1f}]",
                     f"{100 * r['heldout']:+.1f} [{100 * r['heldout_lo']:+.1f}, {100 * r['heldout_hi']:+.1f}]",
                     pct(r["p_heldout_pos"], 0), pct(r["p_heldout_neg"], 0), pct(r["p_base_picked"], 0),
                     f"{r['top_pick']} ({pct(r['top_pick_share'], 0)})",
                     f"{r['full_best']} {100 * r['full_best_gain']:+.1f}"])
    L += md_table(["model", "task", "items", "variants", "base accuracy", "selected gain (pp)", "held-out gain (pp)",
                   "held out > 0", "held out < 0", "base picked", "most picked", "best on all items (pp)"], rows)
    sel = np.array([r["selected"] for r in bp.values() if r])
    held = np.array([r["heldout"] for r in bp.values() if r])
    if len(sel):
        tail = ("so none of the apparent gain carries over to new items" if held.mean() <= 0 else
                f"so about {100 * held.mean() / sel.mean():.0f}% of the apparent gain carries over to new items")
        L.append(f"Across these {len(sel)} cells the mean selected gain is {100 * sel.mean():+.1f} pp and the mean "
                 f"held-out gain {100 * held.mean():+.1f} pp, {tail}.")
        L.append("")

    # 7 meaning
    L += ["## 7. Meaning preservation", "",
          "Templates were written by hand and reviewed; tests check that every variant contains the item verbatim, "
          "that MMLU options appear once each and the key follows a shuffle, and that only the system variant uses "
          "a system message. Two further checks are below: the accuracy change of each variant for every model "
          "(a rephrasing that changed the question should cost accuracy for all of them), and a sample of rendered "
          "prompts.", ""]
    mc = A.meaning_check(common, full or models)
    rows = []
    for r in mc:
        rows.append([TASK_LABEL[r["task"]], r["variant"]] + [pp(r["per_model"][m]) for m in (full or models)]
                    + ["yes" if r["all_down"] else ""])
    L += md_table(["task", "variant"] + [mlabel(m) for m in (full or models)] + ["below zero for every model"], rows)
    L += ["<details><summary>Rendered prompts for one item per task</summary>", ""]
    for t in C.TASKS:
        it = next(i for i in items if i["task"] == t)
        L.append(f"**{TASK_LABEL[t]}, `{it['id']}`**")
        L.append("")
        for v in variants(t):
            r = render(it, v)
            L.append(f"`{v}` (gold in this frame: `{r.gold if t != 'extract' else 'fields'}`)")
            L.append("")
            L.append("```text")
            for msg in r.messages:
                L.append(f"[{msg['role']}]")
                L.append(msg["content"])
            L.append("```")
            L.append("")
    L += ["</details>", ""]

    # 8 parser check
    L += ["## 8. Answer-parser check", ""]
    ag = os.path.join(C.HANDCHECK, "agreement.json")
    if os.path.exists(ag):
        a = json.load(open(ag))
        L += [a["summary"], ""]
    else:
        L += ["Not yet labelled.", ""]
    rows = []
    for m in models:
        for t in C.TASKS:
            d = df[(df.model == m) & (df.task == t)]
            meth = d.method.value_counts(normalize=True)
            rows.append([mlabel(m), TASK_LABEL[t], ", ".join(f"{k} {100 * v:.1f}%" for k, v in meth.items())])
    L += ["Parse rules used over all prompts:", ""] + md_table(["model", "task", "rule"], rows)

    # 9 hypotheses
    L += ["## 9. Pre-registered predictions", "", "Scored against `HYPOTHESES.md` (written before the full run).", ""]
    L += hypothesis_lines(df, models, full, kt, floors, bp, reph, A.rephrased(common))

    with open(os.path.join(C.REPORTS, "results.md"), "w") as f:
        f.write("\n".join(L).rstrip() + "\n")
    print("wrote reports/results.md and", len(os.listdir(C.FIGURES)), "figures")


def hypothesis_lines(df, models, full, kt, floors, bp, reph, creph) -> list[str]:
    out = []
    ms = full or models

    def verdict(ok):
        return "held" if ok else "did not hold"

    # H1: every full-grid model flips on at least 5% of items averaged over rephrasings, above the floor
    vals = {m: item_ci(reph[reph.model == m], "flip") for m in ms}
    ok = all(v["mean"] >= 0.05 and v["lo"] > floors[m]["mean"] for m, v in vals.items())
    out.append("1. Average flip rate of at least 5% for every model, interval above the A/A rate: "
               + verdict(ok) + " (" + "; ".join(f"{mlabel(m)} {ci(v)}" for m, v in vals.items()) + ").")
    # H2: A/A floor under 2%
    ok = all(floors[m]["mean"] < 0.02 for m in ms)
    out.append("2. A/A flip rate under 2% for every model: " + verdict(ok) + " ("
               + "; ".join(f"{mlabel(m)} {ci(floors[m])}" for m in ms) + ").")
    # H3: option order and formatting among the top three kinds by flip rate for most models
    hits = {m: [r["kind"] for r in sorted(kt[m], key=lambda r: -r["mean"])[:3]] for m in ms}
    ok = sum(("option order" in h) for h in hits.values()) > len(ms) / 2
    out.append("3. Option order among the three kinds with the highest flip rate for most models: " + verdict(ok)
               + " (" + "; ".join(f"{mlabel(m)}: {', '.join(h)}" for m, h in hits.items()) + ").")
    # H4: politeness and typo within the floor for most models
    def within(m, k):
        r = next((r for r in kt[m] if r["kind"] == k), None)
        return r is not None and not r["excess"]["lo"] > 0
    ok = sum(within(m, "politeness") and within(m, "typo") for m in ms) > len(ms) / 2
    out.append("4. Politeness and the typo not distinguishable from the A/A repeat for most models: " + verdict(ok)
               + " (" + "; ".join(f"{mlabel(m)}: politeness {'within' if within(m, 'politeness') else 'above'}, "
                                  f"typo {'within' if within(m, 'typo') else 'above'}" for m in ms) + ").")
    # H5: flips are mostly neutral: right-to-wrong and wrong-to-right within 3 pp of each other
    rows = []
    ok = True
    for m in ms:
        d = reph[reph.model == m]
        net = item_ci(d, "dacc")
        rows.append(f"{mlabel(m)} net {pp(net)}")
        ok &= abs(net["mean"]) <= 0.03
    out.append("5. Net accuracy change of rephrasing (wrong to right minus right to wrong) within 3 pp of zero: "
               + verdict(ok) + " (" + "; ".join(rows) + ").")
    # H6: flips concentrate in tasks: GSM8K (long reasoning) has the highest flip rate
    by_task = {t: item_ci(creph[creph.task == t], "flip")["mean"] for t in C.TASKS if (creph.task == t).any()}
    ok = max(by_task, key=by_task.get) == "gsm8k"
    out.append("6. GSM8K has the highest flip rate of the three tasks (main models, common items): " + verdict(ok) + " ("
               + ", ".join(f"{TASK_LABEL[t]} {pct(v)}" for t, v in by_task.items()) + ").")
    # H7: winner's curse: held-out gain under half the selected gain on average, and its middle range covers 0
    cells = [bp[(m, t)] for m in ms for t in C.TASKS if bp.get((m, t))]
    sel = np.mean([c["selected"] for c in cells])
    held = np.mean([c["heldout"] for c in cells])
    cover = sum(c["heldout_lo"] <= 0 <= c["heldout_hi"] for c in cells)
    out.append(f"7. Best-of-N held-out gain less than half the selected gain on average, and the held-out range "
               f"covers zero in most model-task cells: {verdict(held < sel / 2 and cover > len(cells) / 2)} "
               f"(selected {100 * sel:+.1f} pp, held out {100 * held:+.1f} pp, range covers zero in {cover} of "
               f"{len(cells)} cells).")
    # H8: the larger model flips less
    if "qwen3:30b-a3b" in models:
        big = item_ci(reph[reph.model == "qwen3:30b-a3b"], "flip")
        ids = set(reph[reph.model == "qwen3:30b-a3b"].id)
        small = item_ci(reph[(reph.model == "qwen3:8b") & reph.id.isin(ids)
                             & reph.variant.isin(set(reph[reph.model == "qwen3:30b-a3b"].variant))], "flip")
        v = ("not testable on these items: neither model changed an answer" if big["mean"] == small["mean"] == 0
             else verdict(big["hi"] < small["lo"]))
        out.append(f"8. qwen3:30b-a3b (thinking) flips less than qwen3:8b on the same items and variants: {v} "
                   f"({len(ids)} MMLU items; 30b {ci(big)}, 8b {ci(small)}; confounded with thinking).")
    else:
        out.append("8. Larger model flips less: not tested (qwen3:30b-a3b not run).")
    # H9: llama flips most among full-grid models
    fl = {m: item_ci(creph[creph.model == m], "flip")["mean"] for m in ms if m not in C.THINKING_ONLY}
    if "llama3.1:8b" in fl and len(fl) > 1:
        out.append("9. llama3.1:8b has the highest average flip rate of the main models (on the items all completed): "
                   + verdict(max(fl, key=fl.get) == "llama3.1:8b") + " ("
                   + ", ".join(f"{mlabel(m)} {pct(v)}" for m, v in fl.items()) + ").")
    return [x + "\n" for x in out]
