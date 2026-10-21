import pytest

from brittle.data import e2e_usable, gsm8k_gold, norm, parse_mr
from brittle.parse import check, extract_key, parse_extract, parse_gsm8k, parse_mmlu


@pytest.mark.parametrize("text,want,how", [
    ("15 * 1/5 = 3\n15 + 3 = 18\nAnswer: 18", "18", "cue"),
    ("Total is $1,250.\n\n**Answer: $1,250**", "1250", "cue"),
    ("So she pays 18 dollars.\nA: 18", "18", "cue"),
    ("Final Answer: 7.50", "7.5", "cue"),
    ("Answer: 18.00", "18", "cue"),
    ("Answer: The total is 42 apples.", "42", "cue"),
    ("The answer is \\boxed{12}.", "12", "boxed"),
    ("First 3 then 4, so 7 in total.", "7", "last number"),
    ("Answer: -5", "-5", "cue"),
    ("I cannot tell.", None, "none"),
    # the last cue wins, so a restated question number does not
    ("Answer: 3 apples first\nThen more.\nAnswer: 9", "9", "cue"),
])
def test_gsm8k_parser(text, want, how):
    assert parse_gsm8k(text) == (want, how)


def test_gsm8k_gold_and_check():
    assert gsm8k_gold("blah\n#### 1,234") == "1234"
    assert check("gsm8k", "Answer: 1234", "1234")["correct"]
    assert not check("gsm8k", "Answer: 1235", "1234")["correct"]


@pytest.mark.parametrize("text,want,how", [
    ("Answer: B", "B", "cue"),
    ("**Answer: C**", "C", "cue"),
    ("A: D", "D", "cue"),
    ("Answer: A. genetic drift.", "A", "cue"),
    ("Answer: (B) mitochondrion", "B", "cue"),
    ("C", "C", "bare letter"),
    ("(D)", "D", "bare letter"),
    ("B. mitochondrion", "B", "leading letter"),
    ("The correct answer is C because ...", "C", "answer is"),
    ("A: a population shrinks", "A", "leading letter"),
    ("I am not sure about this one.", None, "none"),
])
def test_mmlu_parser(text, want, how):
    assert parse_mmlu(text) == (want, how)


GOLD = {"name": "The Eagle", "eat_type": "coffee shop", "food": "French", "area": "riverside", "near": "Burger King"}


def test_extract_parser_fenced_and_normalised():
    out = ('```json\n{"name": "the eagle", "eat_type": "Coffee Shop", "food": "French", '
           '"area": "Riverside", "near": "Burger King."}\n```')
    a, how = parse_extract(out)
    assert how == "json" and a["eat_type"] == "Coffee Shop"
    r = check("extract", out, GOLD)
    assert r["correct"] and r["fields_correct"] == 5


def test_extract_parser_partial_and_broken():
    out = '{"name": "The Eagle", "eat_type": "restaurant", "food": "French", "area": "riverside", "near": null}'
    r = check("extract", out, GOLD)
    assert not r["correct"] and r["fields_correct"] == 3
    assert check("extract", "name: The Eagle", GOLD)["answer"] is None
    assert check("extract", '{"name": "The Eagle", ', GOLD)["answer"] is None


def test_extract_key_ignores_case_accents_and_punctuation():
    a = {"name": "Café Rouge", "eat_type": "pub", "food": "Fast food", "area": "city centre", "near": "The Bakers"}
    b = {"name": "cafe rouge", "eat_type": "Pub", "food": "fast food.", "area": "City Centre", "near": "the bakers"}
    assert extract_key(a) == extract_key(b)
    assert norm("Café  Sicilia!") == "cafe sicilia"


def test_e2e_item_filter():
    mr = parse_mr("name[The Eagle], eatType[coffee shop], food[French], area[riverside], near[Burger King]")
    assert e2e_usable(mr, "The Eagle is a French coffee shop by the riverside near Burger King.") == GOLD
    # a value missing from the text, or a second area named, rules the pair out
    assert e2e_usable(mr, "The Eagle is a coffee shop by the riverside near Burger King.") is None
    assert e2e_usable(mr, "The Eagle, a French coffee shop by the riverside, not the city centre, "
                          "near Burger King.") is None
    assert e2e_usable(parse_mr("name[The Eagle], eatType[coffee shop]"), "The Eagle coffee shop") is None


OPTS = ["genetic drift", "a population shrinks", "stabilizing selection.", "Phagocytes"]


@pytest.mark.parametrize("text,want,how", [
    ("D: Phagocytes", "D", "option text"),
    # under the "A: <letter>" marker, "A:" followed by another option's text names that option
    ("A: a population shrinks", "B", "option text"),
    ("C: stabilizing selection\n\nExplanation: ...", "C", "option text"),
    ("A: C", "C", "cue"),
    ("A: A. genetic drift.", "A", "cue"),
    ("B: something not among the options", "B", "leading letter"),
])
def test_mmlu_parser_with_option_texts(text, want, how):
    assert parse_mmlu(text, OPTS) == (want, how)
