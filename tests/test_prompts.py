import json
import re

import pytest

from brittle.data import E2E_FIELDS
from brittle.parse import check
from brittle.prompts import KIND, TEMPLATES, permutation, render, to_original, variants

GSM = {"id": "gsm8k-0001", "task": "gsm8k", "split": "main", "source_index": 1,
       "question": "Tom has {3} apples and buys 4 more. How many apples does he have?", "gold": "7"}
MC = {"id": "mmlu-high_school_biology-001", "task": "mmlu", "split": "main", "source_index": 1,
      "subject": "high_school_biology", "question": "Which organelle makes ATP?",
      "options": ["nucleus", "mitochondrion", "ribosome", "vacuole"], "gold": "B"}
EX = {"id": "extract-0001", "task": "extract", "split": "main", "source_index": 1,
      "text": "The Eagle is a French coffee shop in the riverside area near Burger King.",
      "gold": {"name": "The Eagle", "eat_type": "coffee shop", "food": "French", "area": "riverside",
               "near": "Burger King"}}
ITEMS = [GSM, MC, EX]


def full_text(r):
    return "\n".join(m["content"] for m in r.messages)


@pytest.mark.parametrize("item", ITEMS, ids=lambda i: i["task"])
def test_every_variant_keeps_the_item_and_key(item):
    for v in variants(item["task"]) + ["aa"]:
        r = render(item, v)
        text = full_text(r)
        if item["task"] == "gsm8k":
            assert item["question"] in text
            assert r.gold == item["gold"]
            assert "<number>" in text
        elif item["task"] == "mmlu":
            assert item["question"] in text
            for o in item["options"]:
                assert sum(line.endswith(". " + o) for line in text.splitlines()) == 1
            shown = dict(re.findall(r"^([A-D])\. (.*)$", text, re.M))
            assert shown[r.gold] == item["options"]["ABCD".index(item["gold"])]
            assert "high school biology" in text and "letter" in text
        else:
            assert item["text"] in text
            for k in E2E_FIELDS:
                assert k in text
            assert '"coffee shop"' in text and '"city centre"' in text
        assert r.messages[-1]["role"] == "user"


def test_only_system_variant_uses_a_system_message():
    for item in ITEMS:
        for v in variants(item["task"]):
            roles = [m["role"] for m in render(item, v).messages]
            assert roles == (["system", "user"] if v == "system" else ["user"])


def test_each_rephrasing_differs_from_base_and_aa_equals_base():
    for item in ITEMS:
        base = render(item, "base").messages
        assert render(item, "aa").messages == base
        for v in variants(item["task"]):
            if v != "base":
                assert render(item, v).messages != base, v


def test_option_shuffles_remap_the_key():
    for v in ["shuf_rev", "shuf_rand"]:
        r = render(MC, v)
        assert sorted(r.perm) == [0, 1, 2, 3]
        assert all(r.perm[i] != i for i in range(4))
        assert to_original(MC, r.gold, r.perm) == MC["gold"]
        for letter in "ABCD":
            shown = dict(re.findall(r"^([A-D])\. (.*)$", full_text(r), re.M))[letter]
            assert MC["options"]["ABCD".index(to_original(MC, letter, r.perm))] == shown
    assert render(MC, "shuf_rev").gold == "C"
    assert render(MC, "shuf_rand").perm != render(MC, "shuf_rev").perm


def test_random_shuffle_is_deterministic_and_item_specific():
    other = {**MC, "id": "mmlu-high_school_biology-002", "source_index": 2}
    assert permutation(MC, "shuf_rand") == permutation(MC, "shuf_rand")
    perms = {tuple(permutation({**MC, "id": f"x-{i}", "source_index": i}, "shuf_rand")) for i in range(30)}
    assert len(perms) > 3 and (3, 2, 1, 0) not in perms
    assert permutation(other, "base") == [0, 1, 2, 3]


def test_shuffled_gold_is_scored_in_the_shown_frame():
    r = render(MC, "shuf_rev")
    assert check("mmlu", f"Answer: {r.gold}", r.gold)["correct"]
    assert not check("mmlu", "Answer: B", r.gold)["correct"]


def test_every_variant_has_a_kind_and_a_template():
    for task, tmpl in TEMPLATES.items():
        for v in variants(task):
            assert v in tmpl and v in KIND
    assert json.dumps(render(EX, "base").messages)
