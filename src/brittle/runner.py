"""Send every (item, variant) prompt to a model, parse the answer, store one row per call."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config as C
from .llm import CacheMiss, Client
from .parse import check
from .prompts import KIND, render, to_original, variants


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def answers_path(model: str, split: str = "main") -> str:
    name = slug(model) + ("" if split == "main" else f"_{split}")
    return os.path.join(C.RESULTS, "answers", f"{name}.jsonl")


class Progress:
    def __init__(self, total: int, label: str, every: int = 50):
        self.total, self.label, self.every = total, label, every
        self.done, self.t0, self.lock = 0, time.time(), threading.Lock()

    def tick(self, client: Client):
        with self.lock:
            self.done += 1
            if self.done % self.every == 0 or self.done == self.total:
                el = time.time() - self.t0
                print(f"[{self.label}] {self.done}/{self.total} new_calls={client.stats['misses']} "
                      f"hits={client.stats['hits']} elapsed={el:.0f}s", flush=True)


def jobs_for(items: list[dict], only: list[str] | None = None) -> list[tuple[dict, str]]:
    """Item by item, so an interrupted run leaves whole items behind; variant order is shuffled per item
    so the base prompt and its A/A repeat are not sent back to back."""
    jobs = []
    rng = random.Random(C.SEED)
    for it in items:
        vs = [v for v in variants(it["task"]) + ["aa"] if only is None or v in only]
        rng.shuffle(vs)
        jobs += [(it, v) for v in vs]
    return jobs


def run_one(client: Client, model: str, it: dict, v: str) -> dict:
    r = render(it, v)
    out = client.chat(model, r.messages, C.options(it["task"], model), tag="aa" if v == "aa" else "")
    chk = check(it["task"], out["text"], r.gold, r.options)
    ans = to_original(it, chk["answer"], r.perm) if it["task"] == "mmlu" else chk["answer"]
    return {
        "model": model, "id": it["id"], "task": it["task"], "split": it["split"], "variant": v, "kind": KIND[v],
        "output": out["text"], "answer": ans, "correct": bool(chk["correct"]), "method": chk["method"],
        "fields_correct": chk.get("fields_correct"),
        "truncated": out.get("done_reason") == "length",
        "prompt_tokens": out["prompt_tokens"], "output_tokens": out["output_tokens"],
        "seconds": out["seconds"], "cached": out["cached"],
    }


def run(items: list[dict], client: Client, model: str, workers: int = 4, only: list[str] | None = None,
        split: str = "main", skip_missing: bool = False) -> list[dict]:
    jobs = jobs_for(items, only)
    prog = Progress(len(jobs), model)

    def work(job):
        try:
            row = run_one(client, model, *job)
        except CacheMiss:
            if not skip_missing:
                raise
            row = None
        prog.tick(client)
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(work, jobs) if r is not None]
    path = answers_path(model, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = []
    if os.path.exists(path):
        keys = {(r["id"], r["variant"]) for r in rows}
        with open(path) as f:
            old = [r for r in map(json.loads, f) if (r["id"], r["variant"]) not in keys]
    rows = sorted(old + rows, key=lambda r: (r["task"], r["id"], r["variant"]))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows
