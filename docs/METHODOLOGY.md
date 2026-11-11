# Methodology and artifact contracts

## Data and splits

Original paths under `prm800k/data`, `math_splits`, `instructions`, `grading` and
`img` remain available. Original JSONL files in a non-LFS clone are pointers.
The `download` command retrieves immutable upstream-commit assets into `.cache`
and checks their published LFS size and SHA-256. The offline gzip fixtures are
small enough for ordinary Git, and are included in wheels.

The bundled train fixture contains 128 phase-1 and 640 phase-2 usable rows,
selected by seeded reservoir sampling. Test selection is 32 random eligible
problems with at least four usable annotations; all 184 of their candidates
are retained. Selection is not stratified by model success or correctness.
This is not a representative uniform benchmark: phase-2 candidates were
collected through active learning. The full source hashes and selection counts
are in `prm800k/fixtures/manifest.json`. Regenerate with
`python scripts/build_fixtures.py`.

The demo splits training problems into 80% train / 20% validation with a seeded
shuffle of sorted unique problems. Duplicate solutions stay together. The CLI
`train` uses a streaming SHA-256 problem split instead, so full datasets need not
be materialized. Explicit validation files are checked for problem overlap.
Neither test split participates in model selection, acquisition or training.
The original special MATH split is preserved; this is not the conventional MATH
7,500/5,000 split.

Each labeled alternative sees the selected previous trajectory, not its sibling
alternatives. Human replacements count as positive, including the object form
actually present in the release. A null chosen completion without a replacement
ends the prefix. Quality-control, screening, unfinished, flagged and unrated
examples are excluded. Traversal stops after an incorrect selected step.

## Model inputs and training

The small model has byte embeddings, learned positions, two causal transformer
blocks (64 dimensions, four heads, zero dropout) and a classification head.
The default demo has 133,251 PRM parameters and 133,186 ORM parameters. This is
not a pretrained mathematical language model.

Inputs contain only the question and solution prefix. Encoding reserves up to
one third of the context for question bytes and retains the newest solution
bytes. This preserves the scored step endpoint under a strict memory budget,
but drops earlier context for long examples. Context truncation counts are
reported separately for training and validation. Prefixes are scored separately
when sliding context is needed; no future step can influence a prior score.

PRM targets map negative/neutral/positive to 0/1/2. The loss is cross-entropy at
the last token of a labeled step. ORM labels are repeated over all retained
solution tokens, with prompt/padding masked; inference uses the final token.
Training uses AdamW and gradient clipping. Reservoir caps bound the number of
encoded examples held in RAM. MPS runs float32; CPU uses the same architecture.
Seeds are recorded but cross-device bitwise equality is not promised.

`orm-teacher` consumes `teacher_outcome` labels rather than the automatically
checked final answer. Synthetic PRM labels stop at the first teacher prediction
with P(negative) > 0.2. Active acquisition ranks training candidates by their
PRM score, fills floor(0.8 × budget) from wrong answers and fills the remainder
from all unselected candidates. A scarce wrong-answer pool is filled by the
remaining ranking. No selected row is duplicated.

The optional Qwen path uses a fixed, revision-pinned local generator. SFT LoRA
adapters train only on answer-correct candidates; labels mask the prompt.
Math adaptation accepts supplied text. These adapters are separate from the
small reward models and are not automatically used in default demonstrations.
The base model has its own training history; no contamination claim is made.

## Candidate format

Each JSONL record has the following fields; use gzip optionally:

```json
{
  "id": "unique-candidate-id",
  "problem": "What is 1 + 1?",
  "steps": ["Add one and one.", "The result is 2."],
  "answer": "2",
  "ground_truth_answer": "2",
  "is_correct": true
}
```

`score` adds `step_probabilities` in **[negative, neutral, positive]** order,
`prm_score`, the other three PRM reductions, and optionally `orm_score`.
Locally produced PRM scores are **log probabilities**; released PRM scores are
ordinary probabilities. Ranking is valid in either representation, but do not
mix them in one candidate group. The evaluator also accepts the upstream
`given_answer` field. An answerless candidate uses JSON null and stays in the
population as a failed attempt. Empty step arrays are rejected for scoring.

## Exact best-of-N calculation

For a score tier, let T be the attempted generation population, h the number
of usable candidates with strictly higher score, k the tier size, c the number
of correct candidates in that tier, and n the number of attempts drawn without
replacement. The tier's contribution to expected correctness is:

```text
[ C(T-h,n) / C(T,n) - C(T-h-k,n) / C(T,n) ] × c/k
```

Summing over tiers gives the expected pass rate. Missing/answerless generations
are included in T but never in the winning tiers. C(a,n) is zero if a<n. A
cumulative product recurrence computes all avoidance probabilities in O(T)
per N and avoids cancellation from factorials. Ties average uniformly, matching
the first maximum after a random shuffle. Exhaustive enumeration tests cover
all subsets of small populations, ties and missing slots.

Monte Carlo uses one seeded permutation per problem/trial and takes prefixes
for each N. This preserves each N's sampling distribution while pairing draws
across methods and N. It changes cross-N covariance relative to independent
upstream shuffles, not the expected score at each N. Score selection uses
vectorized arrays; majority counts normalized answers and resolves ties by the
first occurrence in the shuffled draw. Oracle is an upper bound, not a deployable
selector. Random picks the first answer-bearing candidate in the draw.

Reported `std` is the standard deviation across resampling trials, not a
confidence interval over possible test questions. `monte_carlo_standard_error`
is std/sqrt(trials). Exact means have no resampling error. Small subset results
should not be compared to full-paper numbers.

## Memory and numerical choices

- JSONL auditing, full training input and official-score import are streamed.
- SQLite stores problem IDs, answer strings, correctness and two scalar scores;
  original solution text and per-step teacher distributions stay out of RAM.
- Evaluation loads one problem at a time. Temporary Monte Carlo arrays scale
  with trials × population, not the number of test problems.
- Log-domain PRM products prevent long-solution underflow. Probability inputs
  must be finite, normalized and within [0,1].
- Training and inference avoid CUDA-only code, mixed-precision assumptions,
  multiprocessing DataLoader workers and compilation requirements.

## Evidence retained

The primary run contains `config.json`, `inputs.json`, `environment.json`,
`data_audit.json`, both `.pt` checkpoints, training metrics, `scored.jsonl.gz`,
active selections, synthetic labels, `evaluation.json`, charts, HTML,
`ANALYSIS.md` and a checksum manifest. Optional runs retain base-model revision,
generation budget and raw generations. Synthetic/active ablations retain their
exact training subsets for each seed/condition. The recording script retains
its actual terminal log.

Full public assets live in ignored `.cache`; compact fixtures and experimental
outputs are versioned. Generated JSONL uses `.jsonl.gz` so the upstream global
`*.jsonl` LFS rule does not hide new evidence behind pointers. Original data
paths and license files are unchanged.
