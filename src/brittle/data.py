"""Download the pinned benchmark files and build the item set deterministically."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

E2E_FIELDS = ["name", "eat_type", "food", "area", "near"]
E2E_MR_KEYS = {"name": "name", "eatType": "eat_type", "food": "food", "area": "area", "near": "near"}
EAT_TYPES = ["restaurant", "pub", "coffee shop"]
AREAS = ["riverside", "city centre"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_path(name: str) -> str:
    return os.path.join(C.RAW, f"{name}.parquet")


def fetch() -> dict:
    os.makedirs(C.RAW, exist_ok=True)
    out = {}
    for name, (repo, rev, path, pinned) in C.SOURCES.items():
        dest = raw_path(name)
        if not os.path.exists(dest):
            url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
        got = sha256(dest)
        if pinned and got != pinned:
            raise RuntimeError(f"{dest}: sha256 {got} does not match pinned {pinned}")
        out[name] = {"repo": repo, "revision": rev, "path": path, "bytes": os.path.getsize(dest), "sha256": got}
    with open(os.path.join(C.DATA, "sources.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def parse_mr(mr: str) -> dict:
    return {k.strip(): v.strip() for k, v in re.findall(r"([^\[,]+)\[([^\]]*)\]", mr)}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def e2e_usable(mr: dict, text: str) -> dict | None:
    """Gold fields for an E2E pair if every field is present and stated verbatim in the text."""
    if not all(k in mr for k in E2E_MR_KEYS):
        return None
    gold = {E2E_MR_KEYS[k]: mr[k] for k in E2E_MR_KEYS}
    if gold["eat_type"] not in EAT_TYPES or gold["area"] not in AREAS:
        return None
    t = " " + norm(text) + " "
    if not all(f" {norm(v)} " in t for v in gold.values()):
        return None
    if norm(gold["name"]) == norm(gold["near"]):
        return None
    # the other eat type or area must not also appear, so the text states one value only
    if sum(f" {norm(e)} " in t for e in EAT_TYPES) != 1 or sum(f" {norm(a)} " in t for a in AREAS) != 1:
        return None
    return gold


def _sample(n_total: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n_total)


def build_items() -> list[dict]:
    """Pilot items first, then main items, per task. All draws are seeded and sorted by source order."""
    items = []
    n = C.N_PER_TASK + C.N_PILOT

    g = pd.read_parquet(raw_path("gsm8k_test")).reset_index(drop=True)
    for rank, i in enumerate(_sample(len(g), C.SEED)[:n]):
        r = g.iloc[i]
        items.append({"id": f"gsm8k-{i:04d}", "task": "gsm8k", "split": "pilot" if rank < C.N_PILOT else "main",
                      "question": r.question.strip(), "gold": gsm8k_gold(r.answer), "source_index": int(i)})

    per_subject = C.N_PER_TASK // len(C.MMLU_SUBJECTS)
    pilot_per = -(-C.N_PILOT // len(C.MMLU_SUBJECTS))
    n_pilot_left = C.N_PILOT
    for si, subj in enumerate(C.MMLU_SUBJECTS):
        m = pd.read_parquet(raw_path(f"mmlu_{subj}")).reset_index(drop=True)
        k_pilot = min(pilot_per, n_pilot_left)
        n_pilot_left -= k_pilot
        order = _sample(len(m), C.SEED + 1 + si)
        for rank, i in enumerate(order[:per_subject + k_pilot]):
            r = m.iloc[i]
            items.append({"id": f"mmlu-{subj}-{i:03d}", "task": "mmlu", "subject": subj,
                          "split": "pilot" if rank < k_pilot else "main",
                          "question": r.question.strip(), "options": [str(c).strip() for c in r.choices],
                          "gold": "ABCD"[int(r.answer)], "source_index": int(i)})

    # interleave subjects so that any prefix of the MMLU items is balanced across them
    mm = [it for it in items if it["task"] == "mmlu"]
    by_subj = [[it for it in mm if it["subject"] == s] for s in C.MMLU_SUBJECTS]
    mixed = [it for row in zip(*by_subj) for it in row]
    mixed += [it for it in mm if it not in mixed]
    items = [it for it in items if it["task"] != "mmlu"] + sorted(mixed, key=lambda it: it["split"] != "pilot")

    e = pd.read_parquet(raw_path("e2e_test")).reset_index(drop=True)
    seen_mr = set()
    usable = []
    for i, r in e.iterrows():
        if r.meaning_representation in seen_mr:
            continue
        gold = e2e_usable(parse_mr(r.meaning_representation), r.human_reference)
        if gold:
            seen_mr.add(r.meaning_representation)
            usable.append((i, r.human_reference.strip(), gold))
    for rank, j in enumerate(_sample(len(usable), C.SEED + 10)[:n]):
        i, text, gold = usable[j]
        items.append({"id": f"extract-{i:04d}", "task": "extract", "split": "pilot" if rank < C.N_PILOT else "main",
                      "text": text, "gold": gold, "source_index": int(i)})
    return items


def write_items() -> list[dict]:
    items = build_items()
    with open(C.ITEMS, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return items


def load_items(split: str | None = "main", tasks: list[str] | None = None) -> list[dict]:
    with open(C.ITEMS) as f:
        items = [json.loads(line) for line in f]
    return [it for it in items if (split is None or it["split"] == split) and (tasks is None or it["task"] in tasks)]
