"""Check of the answer parser against a labelled sample.

`sample` writes handcheck/sample.csv: a seeded random sample of main-run outputs of the models run
without thinking, stratified by task,
with the item and the raw output but without the parser's answer. The labeller reads each output and
fills `label`: the final answer the output commits to, in the prompt's own frame (a number for GSM8K,
a letter as shown in that prompt for MMLU, `name | eat_type | food | area | near` for extraction), or
`none` if it commits to no answer. Saved as handcheck/labels.csv; `score` compares labels with the parser.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import config as C
from .analysis import load_rows
from .data import E2E_FIELDS, load_items, norm
from .parse import canon_number, check, parse_extract
from .prompts import render

COLS = ["id", "subset", "model", "task", "item", "variant", "output"]


def sample(n: int = 50, n_fallback: int = 20, seed: int = 11) -> None:
    """A random sample stratified by task, plus a sample of outputs where the parser needed a rule
    other than the requested format (an `Answer:` line or a JSON object), where mistakes are likelier."""
    os.makedirs(C.HANDCHECK, exist_ok=True)
    df = load_rows("main")
    df = df[~df.model.isin(C.THINKING_ONLY)].reset_index(drop=True)
    per = {t: n // len(C.TASKS) + (1 if i < n % len(C.TASKS) else 0) for i, t in enumerate(C.TASKS)}
    rand = pd.concat([df[df.task == t].sample(n=k, random_state=seed) for t, k in per.items()])
    rand = rand.sample(frac=1, random_state=seed).assign(subset="random")
    rest = df.drop(rand.index)
    fb = rest[~rest.method.isin(["cue", "json"])]
    fb = fb.sample(n=min(n_fallback, len(fb)), random_state=seed).assign(subset="fallback rule")
    s = pd.concat([rand, fb]).reset_index(drop=True).rename(columns={"id": "item"})
    s["id"] = range(1, len(s) + 1)
    s[COLS].assign(label="").to_csv(os.path.join(C.HANDCHECK, "sample.csv"), index=False)
    print(f"wrote {len(s)} rows to handcheck/sample.csv ({len(rand)} random, {len(fb)} fallback)")


def _parser_answer(task: str, output: str, item: dict, variant: str) -> str:
    rd = render(item, variant)
    chk = check(task, output, rd.gold, rd.options)
    if task == "extract":
        a, _ = parse_extract(output)
        return "none" if a is None else " | ".join(norm(a[k]) for k in E2E_FIELDS)
    return "none" if chk["answer"] is None else str(chk["answer"])


def _label_norm(task: str, label: str) -> str:
    label = str(label).strip()
    if label.lower() == "none":
        return "none"
    if task == "gsm8k":
        return canon_number(label) or label
    if task == "mmlu":
        return label.upper()
    return " | ".join(norm(x) for x in label.split("|"))


def score() -> dict:
    lab = pd.read_csv(os.path.join(C.HANDCHECK, "labels.csv"), keep_default_na=False)
    items = {it["id"]: it for it in load_items(None)}
    lab["parser"] = [_parser_answer(r.task, r.output, items[r.item], r.variant) for r in lab.itertuples()]
    lab["label_n"] = [_label_norm(r.task, r.label) for r in lab.itertuples()]
    lab["agree"] = lab.parser == lab.label_n
    out = {"n": int(len(lab)), "agree": int(lab.agree.sum()), "by_subset": {}, "by_task": {}}
    for sub, g in lab.groupby("subset", sort=False):
        out["by_subset"][sub] = {"n": int(len(g)), "agree": int(g.agree.sum())}
    rnd = lab[lab.subset == "random"]
    for t in C.TASKS:
        g = rnd[rnd.task == t]
        out["by_task"][t] = {"n": int(len(g)), "agree": int(g.agree.sum())}
    out["disagreements"] = [{"id": int(r.id), "subset": r.subset, "task": r.task, "label": r.label,
                             "parser": r.parser, "output": r.output[-300:]} for r in lab[~lab.agree].itertuples()]
    parts = [f"{sub}: {v['agree']}/{v['n']}" for sub, v in out["by_subset"].items()]
    tparts = [f"{t} {v['agree']}/{v['n']}" for t, v in out["by_task"].items()]
    out["summary"] = ("A single labelling pass, with no second labeller, read each sampled output without seeing the "
                      "parser's answer and wrote down the answer the output commits to (`none` if it commits to "
                      "none). Agreement with the parser: " + "; ".join(parts)
                      + " (random sample by task: " + ", ".join(tparts) + ").")
    if out["disagreements"]:
        out["summary"] += " Disagreements: " + "; ".join(
            f"#{d['id']} ({d['task']}, {d['subset']}) label `{d['label']}`, parser `{d['parser']}`"
            for d in out["disagreements"]) + "."
    else:
        out["summary"] += " No disagreements."
    with open(os.path.join(C.HANDCHECK, "agreement.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(out["summary"])
    return out
