# Hypotheses and analysis plan

Written 2026-10-20 at 16:33 EDT, before the full run. Nothing below is edited after the full run starts;
corrections go in a dated addendum at the end.

What had been seen when this was written: a pilot on 6 pilot items (2 per task) with `qwen3:8b`,
`llama3.1:8b` and `gemma3:12b`, every variant, used to check the templates and the parsers on real
outputs, plus 10 answer-only GSM8K prompts on two models (which showed that answer-only maths is near
chance for these models, so worked reasoning stays in). The pilot items (10 per task) are excluded from
every reported number. The pilot showed, on far too few items to estimate anything: `llama3.1:8b`
changing its multiple-choice answer under several rephrasings of one question, the other two models
stable on the same items, and every output parseable. Predictions informed by that are marked (P).

## Primary question

When a task prompt is rephrased in ways that should not change its meaning, how often does the parsed
answer change, how often does it change from right to wrong, which kinds of rephrasing matter most, and
is the best-scoring prompt variant a real improvement or a lucky draw?

## Predictions

Flip rate = share of items whose parsed answer differs from the answer under the base prompt. pp =
percentage points. "Floor" = the flip rate when the base prompt is sent a second time.

1. Averaged over all rephrasings, every model's flip rate is at least 5%, with an interval above its
   A/A rate.
2. The A/A flip rate (same prompt, temperature 0, fixed seed) is under 2% for every model.
3. Option order is among the three kinds of rephrasing with the highest flip rate for most models
   (prior work on option-order sensitivity).
4. Politeness and a single typo in the instruction are not distinguishable from the A/A repeat for
   most models.
5. Flips are mostly sideways: the net accuracy change of rephrasing (wrong to right minus right to
   wrong) is within 3 pp of zero for every model. The base prompt is not specially tuned.
6. GSM8K, where the answer comes after a long worked solution, has the highest flip rate of the three
   tasks.
7. Winner's curse for prompts: picking the best of the 11 (13 for MMLU) variants on half the items and
   scoring it on the other half, the held-out gain over the base prompt is less than half the gain seen
   on the selection half, averaged over model-task cells, and the middle 95% of held-out gains across
   random splits covers zero in most cells.
8. `qwen3:30b-a3b` (run with thinking on) flips less than `qwen3:8b` on the same items and variants.
   This is confounded with thinking.
9. (P) `llama3.1:8b` has the highest average flip rate of the models with the full grid.

What would contradict "harmless rephrasings change answers often": average flip rates within the A/A
floor, or flips confined to option shuffles.

## Analysis plan (fixed in advance)

- Unit of resampling: item. All rephrasings of an item, and its A/A repeat, are paired with the same
  item's base prompt. 95% percentile bootstrap, 2,000 draws; pooled over tasks, items are resampled
  within each task.
- Primary metrics: flip rate, right-to-wrong and wrong-to-right rates (shares of all items), accuracy.
  Secondary: per-item accuracy spread (always right, always wrong, mixed), conditional rates, parse-rule
  shares, length-cap hits.
- Floor rule: a kind of rephrasing is said to change answers only if its flip rate minus the A/A flip rate
  on the same items has an interval above zero.
- Kinds: synonyms (2 variants), instruction order (2), formatting (bullets, answer marker), politeness (1),
  typo (1), system placement (1), output-format wording (1), option order (2, multiple choice only).
  An item's value for a kind with two variants is the mean over both.
- Best prompt: per model and task, 2,000 random half splits; the best variant on the selection half
  (ties at random) is scored on both halves against the base prompt. Also pooled over tasks with the 11
  variants that all tasks share. Reported as mean gains with the middle 95% of splits.
- Meaning preservation: hand-written templates, automated checks that each rendered prompt contains the
  item and key intact, a printed sample of rendered prompts, and the per-model accuracy change of every
  variant (a variant that lowers accuracy for every model is flagged as a possible change of meaning).
- Multiple comparisons: about 8 kinds x 4 models. No correction; predictions 1 to 7 are read as
  confirmatory, everything else as exploratory.
- Exclusions: pilot items. For each model, an item is analysed only if every one of its variants was
  answered (relevant if the run is cut short). A failed call is retried; if it still fails the run stops.
- Parser check: one labelling pass, with no second labeller, over a random sample of 50 outputs
  (stratified by task), written without seeing the parser's answer; agreement is published.

## Run settings

- Items: 100 GSM8K test problems, 100 MMLU test questions (25 each from high school biology, geography,
  macroeconomics and psychology), 100 E2E (cleaned) test descriptions whose five fields are all stated
  verbatim, each drawn with a fixed seed.
- Models: `qwen3:8b` (think off), `llama3.1:8b`, `gemma3:12b`, `qwen3:30b-a3b` (think on, reduced grid).
- Temperature 0, seed 7, `num_ctx` 4096, `num_predict` 512 (GSM8K), 128 (MMLU), 160 (E2E); 4096 for the
  thinking model.

Run order, so that a cut leaves the most useful part done: two models at a time, each going MMLU (all
items), E2E (first 50), GSM8K (first 30), then the rest of E2E and GSM8K. The machine is on battery in
low-power mode while this is written (a worked GSM8K answer takes about 13 seconds), so the full grid may
not finish; whatever is cut is listed in an addendum, and items are analysed only when complete.

## Addenda

**2026-10-20, 16:48 EDT (during the run, before any full-run results were read).** The random option
shuffle could draw the same permutation as the reversal (one derangement in nine), which made the two
option-order prompts identical for about one MMLU item in nine (seen as cache hits in the progress log).
The random shuffle now also excludes the reversal. The affected items are re-asked under the new
shuffle; the reversal and every other variant are unchanged.

**2026-10-20, 23:45 EDT (during the run, before any flip or accuracy-by-variant result was read).** The
machine stayed on battery in low-power mode and slept between about 17:10 and 22:50, so the planned grid
cannot finish. Two models ran at a time at about 4 seconds per extraction call and 15 to 20 seconds per
worked GSM8K answer. Cut to: `qwen3:8b` and `llama3.1:8b` only, on all 100 MMLU items, the first 50 E2E
items and the first 20 GSM8K items (in the seeded order, so each is a random subset), every variant.
`gemma3:12b` and `qwen3:30b-a3b` are not run, so predictions 8 (larger model) and the three-model parts of
others cannot be tested; prediction 9 compares the two 8B models. The pooled "all three tasks"
best-prompt analysis uses the unequal task sizes as they are (halves are still drawn within each task).

**2026-10-20, 23:52 EDT (during the run).** The cut above was too large for the battery left (35%,
about an hour). Revised: GSM8K first 10 items for both models, then E2E up to 30 items and, if time
allows, 50. Prediction 6 (GSM8K flips most) and the GSM8K cells of prediction 7 therefore rest on
10 items and are read as exploratory. By this time the MMLU tables for both models had been generated
once to check the report code.

**2026-10-21, 00:05 EDT (during the run).** The machine went onto mains power (low-power mode off), and a
worked GSM8K answer dropped to about 9 seconds. The grid is extended again, in this order: `qwen3:8b` and
`llama3.1:8b` go on to GSM8K 30, E2E 100 and GSM8K 50; `gemma3:12b` runs MMLU 100, E2E 50 and GSM8K 30.
Whatever has finished when the run is stopped is reported, and the final grid is stated in the README
and results. `qwen3:30b-a3b` runs last, on a reduced grid, only if time allows.

**2026-10-21, 00:22 EDT (during the run).** To keep one grid for every model and stop within about two
more hours, the target is now the same for `qwen3:8b`, `llama3.1:8b` and `gemma3:12b`: MMLU 100, E2E 50,
GSM8K 30. The extra E2E 100 and GSM8K 50 steps above are dropped.

**2026-10-21, 00:42 EDT (during the run).** `gemma3:12b` is slower (about 6 seconds per E2E call and
about 20 per worked GSM8K answer with the other models running), so it is cut to GSM8K 20. Each model's
own tables use all of its complete items; tables that compare or pool models use the items all of them
completed.

**2026-10-21, 03:10 EDT (end of the run).** `qwen3:30b-a3b` (thinking) started on MMLU 10 at 01:36 but
could only load once `gemma3:12b` finished, and the machine went back onto battery at about 02:40, after
which a call took minutes. It was stopped with 6 MMLU items complete under every variant (93 calls). Its
results, and prediction 8, rest on those 6 items and are exploratory. Final grid: MMLU 100, E2E 50, GSM8K
30 for `qwen3:8b` and `llama3.1:8b`; MMLU 100, E2E 50, GSM8K 20 for `gemma3:12b`; MMLU 6 for
`qwen3:30b-a3b`.
