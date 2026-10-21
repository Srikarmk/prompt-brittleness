"""Prompt templates: one base prompt per task and one rephrasing per variant.

Each variant changes one thing about the base prompt. Every template is written out in full so it
can be read and checked by eye; `render` fills in the item. Option-order variants (multiple choice
only) keep the base wording and permute the options, remapping the answer key.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C

# variant -> kind. "aa" is the base prompt sent a second time (the A/A noise floor).
KIND = {
    "base": "base",
    "aa": "identity (A/A)",
    "syn1": "synonyms", "syn2": "synonyms",
    "order1": "instruction order", "order2": "instruction order",
    "bullets": "formatting", "marker": "formatting",
    "polite": "politeness",
    "typo": "typo",
    "system": "system placement",
    "outfmt": "output-format wording",
    "shuf_rev": "option order", "shuf_rand": "option order",
}
KINDS = ["synonyms", "instruction order", "formatting", "politeness", "typo", "system placement",
         "output-format wording", "option order"]
COMMON = ["base", "syn1", "syn2", "order1", "order2", "bullets", "marker", "polite", "typo", "system", "outfmt"]
VARIANTS = {
    "gsm8k": COMMON,
    "mmlu": COMMON + ["shuf_rev", "shuf_rand"],
    "extract": COMMON,
}

# (system, user). {body} is the item; {subject} is the MMLU subject.
G_T = "Solve the following math problem."
G_F = 'Reason step by step, then write the final answer on the last line as "Answer: <number>".'
GSM8K = {
    "base": (None, f"{G_T} {G_F}\n\nProblem: {{body}}"),
    "syn1": (None, 'Work out the following math problem. Think step by step, then put the final answer on the '
                   'last line as "Answer: <number>".\n\nProblem: {body}'),
    "syn2": (None, 'Solve the math question below. Reason through it step by step, then write the final result '
                   'on the last line as "Answer: <number>".\n\nQuestion: {body}'),
    "order1": (None, f"Problem: {{body}}\n\nSolve the math problem above. {G_F}"),
    "order2": (None, f"{G_F} {G_T}\n\nProblem: {{body}}"),
    "bullets": (None, "Instructions:\n- Solve the following math problem.\n- Reason step by step.\n"
                      '- Write the final answer on the last line as "Answer: <number>".\n\nProblem: {body}'),
    "marker": (None, f'{G_T} Reason step by step, then write the final answer on the last line as "A: <number>".'
                     "\n\nProblem: {body}"),
    "polite": (None, f"Please solve the following math problem. {G_F} Thank you!\n\nProblem: {{body}}"),
    "typo": (None, f"Solve the folowing math problem. {G_F}\n\nProblem: {{body}}"),
    "system": (f"{G_T} {G_F}", "Problem: {body}"),
    "outfmt": (None, f'{G_T} Reason step by step. Your reply must end with the final answer, written as '
                     '"Answer: <number>" on its own line.\n\nProblem: {body}'),
}

M_T = "Answer the following multiple-choice question about {subject}."
M_F = 'Reply with the letter of the correct option, in the form "Answer: <letter>".'
MMLU = {
    "base": (None, f"{M_T} {M_F}\n\nQuestion: {{body}}"),
    "syn1": (None, "Respond to the following multiple-choice question about {subject}. Give the letter of the "
                   'right option, in the form "Answer: <letter>".\n\nQuestion: {body}'),
    "syn2": (None, "Answer the multiple-choice question below about {subject}. Reply with the letter of the "
                   'correct choice, in the form "Answer: <letter>".\n\nQuestion: {body}'),
    "order1": (None, f"Question: {{body}}\n\nAnswer the multiple-choice question above about {{subject}}. {M_F}"),
    "order2": (None, f"{M_F} {M_T}\n\nQuestion: {{body}}"),
    "bullets": (None, "Instructions:\n- Answer the following multiple-choice question about {subject}.\n"
                      '- Reply with the letter of the correct option.\n- Use the form "Answer: <letter>".'
                      "\n\nQuestion: {body}"),
    "marker": (None, f'{M_T} Reply with the letter of the correct option, in the form "A: <letter>".'
                     "\n\nQuestion: {body}"),
    "polite": (None, f"Please answer the following multiple-choice question about {{subject}}. {M_F} Thank you!"
                     "\n\nQuestion: {body}"),
    "typo": (None, f'{M_T} Reply with the leter of the correct option, in the form "Answer: <letter>".'
                   "\n\nQuestion: {body}"),
    "system": (f"{M_T} {M_F}", "Question: {body}"),
    "outfmt": (None, f'{M_T} Your reply must be "Answer: " followed by the letter of the correct option.'
                     "\n\nQuestion: {body}"),
}
MMLU["shuf_rev"] = MMLU["shuf_rand"] = MMLU["base"]

E_T = "Extract the following fields from the restaurant description: name, eat_type, food, area and near."
E_D = ('eat_type is one of "restaurant", "pub" or "coffee shop", area is one of "riverside" or "city centre", '
       "and near is the place it is close to.")
E_F = "Reply with only a JSON object that has exactly these keys."
EXTRACT = {
    "base": (None, f"{E_T} {E_D} {E_F}\n\nDescription: {{body}}"),
    "syn1": (None, "Pull out the following fields from the restaurant description: name, eat_type, food, area "
                   'and near. eat_type is one of "restaurant", "pub" or "coffee shop", area is one of "riverside" '
                   'or "city centre", and near is the place it is next to. Respond with only a JSON object that '
                   "has exactly these keys.\n\nDescription: {body}"),
    "syn2": (None, "Extract these fields from the restaurant text below: name, eat_type, food, area and near. "
                   f"{E_D} Reply with just a JSON object containing exactly these keys.\n\nText: {{body}}"),
    "order1": (None, "Description: {body}\n\nExtract the following fields from the restaurant description above: "
                     f"name, eat_type, food, area and near. {E_D} {E_F}"),
    "order2": (None, f"Reply with only a JSON object that has exactly the keys below. {E_T} {E_D}"
                     "\n\nDescription: {body}"),
    "bullets": (None, f"Instructions:\n- {E_T}\n"
                      '- eat_type is one of "restaurant", "pub" or "coffee shop".\n'
                      '- area is one of "riverside" or "city centre".\n'
                      "- near is the place it is close to.\n"
                      f"- {E_F}\n\nDescription: {{body}}"),
    "marker": (None, f"{E_T} {E_D} {E_F}\n\n### Description\n{{body}}"),
    "polite": (None, f"Please extract the following fields from the restaurant description: name, eat_type, food, "
                     f"area and near. {E_D} {E_F} Thank you!\n\nDescription: {{body}}"),
    "typo": (None, "Extract the following fields from the restaurant descripton: name, eat_type, food, area and "
                   f"near. {E_D} {E_F}\n\nDescription: {{body}}"),
    "system": (f"{E_T} {E_D} {E_F}", "Description: {body}"),
    "outfmt": (None, f"{E_T} {E_D} The output should be a single JSON object with exactly these keys and "
                     "nothing else.\n\nDescription: {body}"),
}

TEMPLATES = {"gsm8k": GSM8K, "mmlu": MMLU, "extract": EXTRACT}
LETTERS = "ABCD"


def variants(task: str) -> list[str]:
    return VARIANTS[task]


def permutation(item: dict, variant: str) -> list[int]:
    """perm[new_position] = original option index."""
    n = len(item.get("options", []))
    if variant == "shuf_rev":
        return list(range(n))[::-1]
    if variant == "shuf_rand":
        # seeded per item; redrawn until no option keeps its place and it differs from the reversal
        rng = np.random.default_rng([C.SEED, int(item["source_index"]), sum(map(ord, item["id"]))])
        while True:
            p = [int(x) for x in rng.permutation(n)]
            if all(p[i] != i for i in range(n)) and p != list(range(n))[::-1]:
                return p
    return list(range(n))


@dataclass
class Rendered:
    messages: list[dict]
    gold: object          # gold answer as the model should give it in this variant
    perm: list[int]       # for multiple choice: perm[shown position] = original index
    options: list[str] | None = None  # option texts in the order shown


def subject_name(item: dict) -> str:
    return item["subject"].replace("_", " ")


def render(item: dict, variant: str) -> Rendered:
    task = item["task"]
    v = "base" if variant == "aa" else variant
    system, user = TEMPLATES[task][v]
    perm: list[int] = []
    opts = None
    if task == "gsm8k":
        body, gold = item["question"], item["gold"]
    elif task == "mmlu":
        perm = permutation(item, v)
        opts = [item["options"][j] for j in perm]
        body = item["question"] + "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        gold = LETTERS[perm.index(LETTERS.index(item["gold"]))]
    else:
        body, gold = item["text"], item["gold"]
    fill = {"body": body, "subject": subject_name(item) if task == "mmlu" else ""}
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system.format(**fill)})
    msgs.append({"role": "user", "content": user.format(**fill)})
    return Rendered(msgs, gold, perm, opts)


def to_original(item: dict, letter: str | None, perm: list[int]) -> str | None:
    """Map a letter in a shuffled prompt back to the original option letter."""
    if letter is None or not perm:
        return letter
    return LETTERS[perm[LETTERS.index(letter)]]
