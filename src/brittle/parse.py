"""Deterministic answer checking.

Each parser returns (answer, method). `answer` is None when nothing usable was found; `method`
records which rule fired, so the report can show how often the fallbacks were needed.
"""

from __future__ import annotations

import json
import re

from .data import E2E_FIELDS, norm

NUM = r"-?\d[\d,]*(?:\.\d+)?"
CUE = r"(?i:final answer|answer|a)"


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("**", "").replace("\\(", "").replace("\\)", "")


def canon_number(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        x = float(s)
    except ValueError:
        return None
    if x == int(x):
        return str(int(x))
    return f"{x:.6g}"


def parse_gsm8k(text: str) -> tuple[str | None, str]:
    t = _clean(text)
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", t)
    cues = list(re.finditer(rf"(?m)^\s*[#>*\- ]*{CUE}\s*[:：]\s*(.*)$", t))
    if cues:
        tail = cues[-1].group(1)
        m = re.search(NUM, tail.replace("$", ""))
        if m:
            return canon_number(m.group(0)), "cue"
    if boxed:
        m = re.search(NUM, boxed[-1])
        if m:
            return canon_number(m.group(0)), "boxed"
    nums = re.findall(NUM, t.replace("$", ""))
    if nums:
        return canon_number(nums[-1]), "last number"
    return None, "none"


def parse_mmlu(text: str, options: list[str] | None = None) -> tuple[str | None, str]:
    """`options` are the option texts in the order shown, used when a reply names an option by its text."""
    t = _clean(text).strip()
    m = re.match(r"\(?([A-D])\)?\s*[:.)]\s*(.+)", t, re.S)
    if m and options:
        # "B: Phagocytes" or, under the "A: <letter>" marker, "A: <text of option C>"
        rest = norm(m.group(2).split("\n")[0])
        hits = [(len(norm(o)), i) for i, o in enumerate(options) if norm(o) and rest.startswith(norm(o))]
        if hits:
            return "ABCD"[max(hits)[1]], "option text"
    cues = re.findall(rf"(?m)^\s*[#>*\- ]*{CUE}\s*[:：]\s*\(?([A-D])\b", t)
    if cues:
        return cues[-1], "cue"
    m = re.search(r"\b(?i:answer|option|choice) is\s*:?\s*\(?([A-D])\b", t)
    if m:
        return m.group(1), "answer is"
    m = re.fullmatch(r"\(?([A-D])\)?[.:)]?", t)
    if m:
        return m.group(1), "bare letter"
    m = re.match(r"\(?([A-D])[.):]\s", t + " ")
    if m:
        return m.group(1), "leading letter"
    m = re.search(r"\b(?i:answer)\s*[:：]?\s*\(?([A-D])\b", t)
    if m:
        return m.group(1), "inline cue"
    return None, "none"


def _json_objects(t: str) -> list[str]:
    out, depth, start = [], 0, None
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(t[start:i + 1])
    return out


def parse_extract(text: str) -> tuple[dict | None, str]:
    t = _clean(text)
    for blob in _json_objects(t):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in E2E_FIELDS):
            return {k: ("" if obj.get(k) is None else str(obj.get(k))) for k in E2E_FIELDS}, "json"
    return None, "none"


def extract_key(ans: dict | None) -> str | None:
    """Canonical string of the normalised fields, used to compare answers across prompts."""
    if ans is None:
        return None
    return json.dumps({k: norm(ans[k]) for k in E2E_FIELDS}, sort_keys=True)


def check(task: str, text: str, gold, options: list[str] | None = None) -> dict:
    """Parse one output. `gold` is the gold answer in the prompt's own frame (shuffled letter for
    reordered options). Returns answer (comparable string), correct, method, and field matches."""
    if task == "gsm8k":
        a, how = parse_gsm8k(text)
        g = canon_number(str(gold))
        return {"answer": a, "correct": a is not None and a == g, "method": how}
    if task == "mmlu":
        a, how = parse_mmlu(text, options)
        return {"answer": a, "correct": a == gold, "method": how}
    a, how = parse_extract(text)
    fields = {k: (a is not None and norm(a[k]) == norm(gold[k])) for k in E2E_FIELDS}
    return {"answer": extract_key(a), "correct": all(fields.values()), "method": how,
            "fields_correct": sum(fields.values())}
