# prompt-brittleness

<a href="https://www.srikarmk.dev/projects/prompt-brittleness"><img alt="Case study" src="https://img.shields.io/badge/case%20study-srikarmk.dev-5b4fc9" /></a> <a href="https://www.srikarmk.dev/series/measuring-llm-systems"><img alt="Series" src="https://img.shields.io/badge/series-Measuring%20LLM%20Systems-5b4fc9" /></a> <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue" /></a> <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue" /></a>

**Question.** Take tasks with checkable answers and rephrase the prompt in ways that should not matter: a
synonym, a reordered instruction, bullets instead of prose, "please", one typo, shuffled options, the
instruction moved to the system message. How often does the answer change, how often from right to
wrong, which kinds of rephrasing matter most, and is the best-scoring prompt a real improvement or a
lucky draw?

This repository is a measurement study with local open-weight models. Every number in
[reports/results.md](reports/results.md) is produced by the code here from cached model calls, so it
regenerates exactly. Pre-registered predictions, with dated addenda: [HYPOTHESES.md](HYPOTHESES.md).

## Findings

Numbers are from [reports/results.md](reports/results.md): 95% bootstrap intervals over items, `pp` =
percentage points. Grid: `qwen3:8b`, `llama3.1:8b` and `gemma3:12b` on 100 GSM8K, 100 MMLU and 100 E2E
items each; `qwen3:30b-a3b` with thinking on, on the 100 MMLU items. Every item under the base prompt, an
A/A repeat and 10 rephrasings (12 for MMLU). A first run was cut short (see Limits); this is the full grid,
run a week later, and the numbers below replace the first run's.

- **Answers move, a little, often.** Averaged over all rephrasings, the parsed answer changed on 4.7%
  [2.9, 6.4] of items for `qwen3:8b`, 7.8% [5.9, 9.8] for `llama3.1:8b` and 4.8% [3.2, 6.6] for
  `gemma3:12b`. Per item that adds up: 16%, 27% and 18% of items gave more than one distinct answer across
  the prompts, and 15%, 24% and 17% were right under some prompts and wrong under others.
- **The noise floor is small, not zero.** Sending the base prompt a second time (temperature 0, fixed seed)
  changed 1, 2 and 2 of 300 answers, all of them worked GSM8K solutions; it changed none of the MMLU or E2E
  answers. Every kind of rephrasing is compared with this repeat on the same items.
- **Option order matters most.** Shuffling the MMLU options (key remapped) changed 11.0% [5.5, 16.5],
  21.0% [14.0, 28.0] and 13.0% [7.5, 18.5] of answers. Next came instruction order (6.8%, 8.8%, 8.0%) and,
  for `llama3.1:8b`, moving the instruction into the system message (9.7%; 4.7% for `qwen3:8b` and 3.7% for
  `gemma3:12b`, whose template has no system role). Politeness and a single typo moved 2 to 6% of answers.
  Putting the E2E description before the instruction cost all three models accuracy (-10, -4 and -16 pp;
  the `qwen3:8b` and `gemma3:12b` intervals exclude zero).
- **Mostly sideways, not down.** Right to wrong and wrong to right were about equal: 1.8% vs 1.8%, 2.1% vs
  2.9%, 2.3% vs 1.2% of items, a net change of -0.0, +0.7 and -1.1 pp, all with intervals touching zero.
  The asymmetry is in the conditional rates: an item the base prompt got right went wrong under a
  rephrasing 2.0 to 2.5% of the time; an item it got wrong was fixed 16 to 20% of the time.
- **The best prompt is a lucky draw.** Picking the best of 11 to 13 wordings on half the items and scoring
  it on the other half, the winner looked +2.6 pp better than the base prompt on the half it was picked on
  and was -0.2 pp on the held-out half, averaged over 12 model-task cells; in all 9 single-task cells the
  middle 95% of held-out gains covers zero. Example: for `llama3.1:8b` on MMLU, the picked wording looked
  +5.5 pp better and held out at +0.6 pp [-6.0, +4.0 over splits].
- **Larger model, thinking on.** On the 100 MMLU items `qwen3:30b-a3b` was right on 95% under the base
  prompt but changed its answer about as often as `qwen3:8b` on the same items: 5.0% [2.1, 8.6] against
  4.7% [2.4, 7.4]. Its own noise floor is higher (the repeat changed 3 of 100 answers), so only option order
  (7.5%) rises clearly above it.
- **Parser check.** A single labelling pass (no second labeller) over 50 random outputs and 20 outputs
  that needed a fallback parsing rule, drawn from the first run, agreed with the parser on 50 of 50 and
  20 of 20.

How the pre-registered predictions fared (details in results.md): held: A/A under 2% (2), option order
among the top three kinds (3), net change within 3 pp (5), GSM8K flipping most (6; 8.0% against 5.9% for
MMLU and 3.4% for E2E), the winner's curse (7), `llama3.1:8b` flipping most (9). Did not hold: an average
flip rate of at least 5% for every model (1; two models were at 4.7 and 4.8%), politeness and typo within
the floor (4), and the 30B thinking model flipping less than `qwen3:8b` (8). The predictions were written
before the first run and committed with its results on 2026-10-21; they were not changed for the full
run, but the first run's results, which cover 180 of these items, had been read by then.

What this adds to prior work: per-item flip direction and an explicit held-out test of the best wording,
on zero-shot chat prompts with a measured noise floor. Sclar et al. and Mizrahi et al. show that scores
move across formats and paraphrases; here the wording that looks best on a dev half does not stay best,
so a single "best prompt" reported from a small dev set is mostly selection.

## Design

**Items.** Three tasks with exact answers, drawn with a fixed seed from public test sets:

| task | source | items sampled (analysed) | answer checked by |
|---|---|---|---|
| maths | GSM8K test | 100 (100) | the number after the last `Answer:` (or `A:`) line, else a `\boxed{}` value, else the last number |
| multiple choice | MMLU test, 25 each from high school biology, geography, macroeconomics, psychology | 100 (100, also on `qwen3:30b-a3b`) | the letter after `Answer:`, else "the answer is X", a bare letter or a leading "X." |
| extraction | E2E NLG (cleaned) test: restaurant descriptions | 100 (100) | a JSON object whose five fields (`name`, `eat_type`, `food`, `area`, `near`) all match after case, accent and punctuation folding |

E2E pairs a meaning representation with a human-written description. An item is kept only if all five
fields are in the representation, each value appears verbatim in the description, and only one eat type
and one area are named, so every gold field is stated in the text. One description per representation.
Ten more items per task were the pilot (used to check templates and parsers on real outputs) and are
excluded from every number.

**Rephrasings.** One base prompt per task and one change at a time, all written out in
[src/brittle/prompts.py](src/brittle/prompts.py):

| kind | variants | what changes |
|---|---|---|
| identity (A/A) | `aa` | nothing: the base prompt sent a second time, as a separate call (the noise floor) |
| synonyms | `syn1`, `syn2` | a few instruction words swapped for synonyms ("Solve" to "Work out", "following" to "below", "Problem:" to "Question:") |
| instruction order | `order1`, `order2` | the item before the instruction; the instruction's sentences in reverse order |
| formatting | `bullets`, `marker` | the instruction as a bulleted list; the answer marker `Answer:` shortened to `A:` (for extraction, which has no answer marker, the `Description:` label becomes a `### Description` heading) |
| politeness | `polite` | "Please ..." and "Thank you!" |
| typo | `typo` | one misspelt instruction word ("folowing", "leter", "descripton"); the item itself is never touched |
| system placement | `system` | the instruction moved, word for word, into the system message |
| output-format wording | `outfmt` | the output-format sentence reworded ("Your reply must end with ...") |
| option order | `shuf_rev`, `shuf_rand` | MMLU only: options reversed, or shuffled per item so no option keeps its place; the key is remapped |

Every rendered prompt is checked by tests: the item text appears verbatim, each MMLU option appears once
and the remapped key points at the right text, only the system variant has a system message, and every
variant differs from the base. A sample of rendered prompts is printed in the report, and the report lists
each variant's accuracy change for every model, since a rephrasing that changed the question would cost
all of them.

**Measures.** For each model, item and variant: the parsed answer and whether it is right. Then:

- *flip rate*: the share of items whose answer differs from the base prompt's answer (no parseable answer
  counts as an answer);
- *right to wrong* and *wrong to right*: shares of all items whose correctness changed in each direction;
- *accuracy spread*: per item, whether it is right under every prompt, none, or some; per model, the
  worst and best variant's accuracy;
- *best prompt*: split the items at random into halves 2,000 times; on each split pick the variant with the
  highest accuracy on the first half, as one would on a dev set, and measure its gain over the base prompt
  on both halves. The gap between the two is the winner's curse: the part of the apparent gain that came
  from picking the luckiest variant.

**Statistics.** Intervals are 95% percentile bootstraps over items (2,000 draws), stratified by task when
tasks are pooled; every comparison is paired by item. A kind of rephrasing counts as changing answers only
when its flip rate minus the A/A flip rate on the same items has an interval above zero. The bootstrap and
A/A rule follow agent-interface-lab and memory-rot in the same series, resampling items instead of users.

**Answer-parser check.** One labelling pass, with no second labeller, read 50 randomly sampled outputs
without seeing the parser's answer and wrote down the answer each commits to; the agreement is in the
report.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) with the models pulled.

```sh
uv sync
uv run brittle fetch          # pinned Hugging Face files, sha256-checked (under 1 MB)
uv run brittle items          # data/items.jsonl: 300 main items and 30 pilot items, seeded
uv run brittle run --models qwen3:8b --split pilot --per-task 2   # pilot: read outputs before the full run
# the full grid (every variant of every main item); the cache skips calls already made
uv run brittle run --models qwen3:8b
uv run brittle run --models llama3.1:8b
uv run brittle run --models gemma3:12b
uv run brittle run --models qwen3:30b-a3b --tasks mmlu
uv run brittle show gsm8k-1248                                       # one item's prompts and outputs
uv run brittle handcheck sample   # then label handcheck/labels.csv and run: brittle handcheck score
uv run brittle report             # reports/results.md and reports/figures/
uv run pytest
```

Every model call is cached in `cache/llm.sqlite`, keyed by model, messages, options, think flag and a
repeat tag (the A/A repeat has its own entry, so it is a real second call). With the cache present,
`brittle run ... --cache-only` rebuilds every row in `results/answers/` without a server, and `report`
re-parses every stored output, so changing a parser rule updates every number. `--per-task N` limits a
run to the first N items of each task in seeded order, which is how the first run was cut.

Runtime: 12,800 model calls in the grid, plus 228 for the pilot. The first run (2026-10-20 16:34 to
2026-10-21 03:10) was mostly on battery in low-power mode, at about 19 tokens a second for one 8B model,
and lost about five and a half hours to the machine sleeping, so it finished only part of the grid. The
full grid was completed from the cache on mains power, from 2026-10-28 21:40 to 2026-10-29 16:38; the
thinking model, at about 10 seconds and 960 output tokens per MMLU answer, took most of that, including a
pause of about nine hours partway through.

## Data and models

| Item | Source | Version | Size | Licence |
|---|---|---|---|---|
| GSM8K test | Hugging Face `openai/gsm8k`, `main/test-00000-of-00001.parquet` | commit `740312add88f781978c0658806c59bc2815b9866` | 419,088 bytes | MIT |
| MMLU test (4 subjects) | Hugging Face `cais/mmlu`, `<subject>/test-00000-of-00001.parquet` | commit `c30699e8356da336a370243923dbaf21066bb9fe` | 238,437 bytes in all | MIT |
| E2E NLG (cleaned) test | Hugging Face `tuetschek/e2e_nlg_cleaned`, `default/test/0000.parquet` (the hub's parquet conversion) | `refs/convert/parquet` commit `6f77a74f6ce3095770924452f3f1b9631f837535` | 222,299 bytes | CC BY-SA 4.0 |
| Models | Ollama 0.40.6 library tags `qwen3:8b`, `llama3.1:8b`, `gemma3:12b`, `qwen3:30b-a3b` (Q4_K_M) | digests in `results/models.json` | 4.9 to 17.7 GB | Apache 2.0 (Qwen3), Llama 3.1 Community Licence, Gemma Terms of Use |

Downloads total under 1 MB; sha256 values are pinned in `src/brittle/config.py` and recorded in
`data/sources.json`. Settings: temperature 0, seed 7, `num_ctx` 4096, at most 512 new tokens for GSM8K,
128 for MMLU and 160 for E2E; `think: false` for `qwen3:8b` (Ollama's `think` request field) and
`think: true` with a 4,096-token cap for `qwen3:30b-a3b`, the thinking-only release, which keeps reasoning
in its visible answer when thinking is turned off (found in memory-rot); only its final answer is parsed.
`gemma3:12b`'s chat template has no system role: Ollama sends a system message to it as a separate user
turn, so its "system placement" variant is a two-message user prompt.

## Layout

```
src/brittle/   config, data (fetch, item sampling), prompts (templates, option shuffles), parse (answer
               checking), llm (Ollama + disk cache), runner, stats (item bootstrap, best of N), analysis,
               report, handcheck, cli
tests/         template rendering, option remapping, parsers on hand cases, cache, bootstrap, best of N
data/          items.jsonl (the sampled items) and sources.json (files, revisions, sha256)
results/       answers/<model>.jsonl (one row per call) and models.json (digests, settings)
reports/       results.md and figures/
handcheck/     sample.csv (drawn by seed), labels.csv (the labelling pass), agreement.json
cache/         llm.sqlite: every model call, keyed by model, messages, options, think flag and repeat tag
```

## Limits

- **Two runs.** The first run was cut short when the machine slept on battery (MMLU 100, E2E 50 and
  GSM8K 20 to 30 items, and 6 items for the 30B model); its cuts are dated in `HYPOTHESES.md`. The full
  grid was finished a week later from the same seeded items and cached calls, after the first run's
  results had been read and published. The predictions were not changed in between.
- The 30B thinking model ran on MMLU only, so its comparison with `qwen3:8b` rests on 100 items.
- Small local models, 4-bit quantised, at temperature 0 and one seed. A hosted model, sampling, or a
  different quantisation could be more or less stable.
- The A/A repeat changed 1 or 2 of 300 answers per main model, all worked GSM8K solutions, and 3 of 100
  for the thinking model. With so low a floor the rule is permissive (any kind with a handful of flips
  clears it), so the sizes matter more than the stars.
- One hand-written template per variant. Each kind is one or two concrete rewrites, not a sample of all
  rewrites of that kind, so "politeness" means these two sentences, not politeness in general.
- Exact-match scoring. In E2E, 28% of the changed answers differ only by a generic word ("English food"
  against "English"); they count as flips and, when the gold is the shorter span, as wrong.
- The `A:` marker collides with option letter A in multiple choice. That is part of what it tests, and the
  parser reads a reply such as "B: Phagocytes" by matching the option text, but some replies to that
  variant stay ambiguous.
- `qwen3:30b-a3b` ran with thinking on, so "larger model" and "reasons before answering" are confounded.
- The parser check is one labelling pass with no second labeller, by the person who also wrote the parser;
  the labeller did not see the parser's answers, but knew its rules. Most outputs are clear-cut, so full
  agreement says the parser does not miss obvious cases, not that it is right on every row.
- Server timings include stretches where two or three models shared the GPU and where the machine was on
  battery; they are not benchmarks.

## Prior work

- Sclar, Choi, Tsvetkov and Suhr. *Quantifying Language Models' Sensitivity to Spurious Features in Prompt
  Design or: How I learned to start worrying about prompt formatting.* ICLR 2024
  ([arXiv:2310.11324](https://arxiv.org/abs/2310.11324)). Meaning-preserving format changes (separators,
  casing, spacing) move few-shot accuracy by up to 76 points for LLaMA-2-13B; the spread persists with
  size, more examples and instruction tuning. Their FormatSpread reports the range over sampled formats.
  This study uses instruction-level rewrites in a zero-shot chat setting and adds per-item flip direction
  and a held-out test of the best format.
- Mizrahi, Kaplan, Malkin, Dror, Shahaf and Stanovsky. *State of What Art? A Call for Multi-Prompt LLM
  Evaluation.* TACL 2024 ([ACL Anthology](https://aclanthology.org/2024.tacl-1.52/)). Instruction
  paraphrases change absolute scores and model rankings across 20 models and 39 tasks; they argue for
  evaluating over many prompts.
- Pezeshkpour and Hruschka. *Large Language Models Sensitivity to The Order of Options in Multiple-Choice
  Questions.* Findings of NAACL 2024 ([arXiv:2308.11483](https://arxiv.org/abs/2308.11483)). Reordering
  options moves accuracy substantially, including on MMLU subsets.
- Zheng, Zhou, Meng, Zhou and Huang. *Large Language Models Are Not Robust Multiple Choice Selectors.*
  ICLR 2024 ([arXiv:2309.03882](https://arxiv.org/abs/2309.03882)). Attributes option-order sensitivity
  mainly to a prior over option letters (token bias) and debiases it with permutations (PriDe).
- Zhu et al. *PromptRobust* (first released as *PromptBench*): *Towards Evaluating the Robustness of Large
  Language Models on Adversarial Prompts* ([arXiv:2306.04528](https://arxiv.org/abs/2306.04528)).
  Character-, word- and sentence-level perturbations of instructions, chosen adversarially; the typo and
  synonym variants here are single, fixed, non-adversarial versions of those.
- Yin, Wang, Horio, Kawahara and Sekine. *Should We Respect LLMs? A Cross-Lingual Study on the Influence of
  Prompt Politeness on LLM Performance.* SICon 2024 ([ACL Anthology](https://aclanthology.org/2024.sicon-1.2)).
  Impolite prompts often hurt; extra politeness does not reliably help.
- He et al. *Does Prompt Formatting Have Any Impact on LLM Performance?*
  ([arXiv:2411.10541](https://arxiv.org/abs/2411.10541), 2024). Plain text, Markdown, JSON and YAML
  templates change GPT-3.5 and GPT-4 results, with no single best format.
- Mirzadeh et al. *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language
  Models* ([arXiv:2410.05229](https://arxiv.org/abs/2410.05229), 2024). Varies the problem itself (names,
  numbers, an irrelevant clause); this study leaves the problem untouched and varies only the instruction.
- Datasets: Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K,
  [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)); Hendrycks et al., *Measuring Massive Multitask
  Language Understanding* (MMLU, ICLR 2021, [arXiv:2009.03300](https://arxiv.org/abs/2009.03300));
  Dušek, Howcroft and Rieser, *Semantic Noise Matters for Neural Natural Language Generation* (INLG 2019,
  the cleaned E2E set, [ACL Anthology](https://aclanthology.org/W19-8652)).

The selection analysis follows Experiment Forensics in the same series, which showed the winner's curse
for A/B test winners; here the "variants" are prompt wordings and the dev set is half the items.

## License

MIT, Kameswara Sai Srikar Manda, 2026. GSM8K and MMLU are MIT-licensed; the E2E cleaned data is
CC BY-SA 4.0. `data/items.jsonl` holds the sampled items (100 E2E descriptions with their fields) under
their original licences.
