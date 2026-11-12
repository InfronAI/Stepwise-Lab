# Paper coverage audit

Reference: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), v1,
2023-05-31. Upstream repository pinned to
`7ecc794703b2877f63226f2477a49b34f9b25163`.

**This is not a 100% reproduction of the original paper.** The original release
is primarily a dataset, grading code and evaluation script. This extension fills
in the runnable methods and records local experiments, but cannot manufacture
unreleased weights, datasets or the original compute budget.

## Implementation and experimental coverage

| Paper component | Original repository | Added implementation | Evidence / limitation |
|---|---|---|---|
| Human process annotations | Released as LFS data | Verified download, streaming audit, real offline fixture | Source hashes and subset-selection recipe in `prm800k/fixtures/manifest.json` |
| Alternative steps and human replacement | Raw schema only | Correct chosen-history expansion, flagged/unrated filtering, object/string human replacements | Semantic tests plus full official-data audit |
| Interactive phase 1 / first-error phase 2 | Released records and instructions | Explicit review state machine, deterministic replay, three paired protocol experiments | 952 official records replayed; 18 MPS training runs plus CPU fallback. Human post-error labels are unavailable; full-trajectory comparison uses identified synthetic labels only |
| Three-class process supervision | No trainer | Causal endpoint loss, positive/neutral/negative targets | MPS and CPU checkpoints and actual loss histories |
| Outcome supervision | No trainer | All-solution-token loss, final-token scoring | Local ORM trained and evaluated; subset is biased, not the original uniform ORM corpus |
| Fixed generator | Scored outputs released | Released candidate replay and local Qwen generation | Pinned base revision; all generations retained, including failures |
| Correct-solution generator SFT | Not released | Answer-filtered, prompt-masked LoRA SFT | Four-example M1 smoke run; Qwen adapters are a local substitute |
| Math-domain adaptation | Corpus and weights absent | LoRA text adaptation on supplied data | Two-example smoke run; no claim to contain MathMix |
| Four PRM aggregation variants | Only final scores used | Product/minimum, neutral included/excluded | Log-domain scores, numerical tests and local comparison |
| Best-of-N selection | Remote-only, unseeded, loads all rows | Streamed disk index, exact expectation, paired Monte Carlo | Exhaustive tiny-population tests; full-score replay stored separately |
| Majority voting | Not implemented | Normalized-answer counts with random ties | Full-release normalized voting 69.20% vs published 69.6%; not an exact match |
| Synthetic process/outcome supervision | No code | Thresholded teacher labels and matched student training | Three seeds, same candidates, local teacher; no strong teacher-quality claim |
| Active data selection | No code | Per-problem/global acquisition; equal-budget uniform comparison | Executed three-seed local ablation; does not establish 2.6× data efficiency |
| Difficulty/domain evaluation | Not implemented | Subject/level and generator-difficulty quintiles; accepts separate benchmark inputs | Paper's complete OOD benchmark data and contamination analysis are not reproduced |
| Original large-scale training/results | Not released | Public-score replay only | Replaying scores is not reproducing original model training |

## What remains unavailable

- The exact GPT-4 base, fine-tuned generator, PRM and ORM checkpoints.
- The private MathMix corpus and its original preprocessing/decontamination pipeline.
- The original uniformly sampled ORM training corpus and all small-model training
  pools/teacher predictions used for the paper's scaling experiments.
- The full original out-of-distribution evaluation setup and human-labeler operations.
- Original-scale training compute, exact internal tokenization and initialization.

There is no defensible way to replace these with a tiny open model and label the
result “100% reproduced.” The included open-model and byte-transformer workflows
are explicitly identified substitutes. No learned verifier here provides a
formal mathematical correctness guarantee.

## Preserved ideas

Step-level supervision remains distinct from answer-level supervision. Both
reward models see causal solution context; neither is given the reference answer
as an input. Candidate generation is fixed during a comparison. PRM scores combine
step predictions, and evaluation measures selection of a correct final answer.
The default scoring policy includes neutral labels as acceptable. Missing
attempts remain part of the evaluation population. Generator RL was outside the
paper's stated scope and is not introduced here.

## Results interpretation

The small model is deliberately cheap enough to run in under a minute on the
recorded M1 Pro. Its context is byte-based and severely truncated for many long
solutions. It is trained from scratch, so a three-class output head is not a
replacement for mathematical pretraining. The local PRM underperforms the ORM
on the recorded subset. ORM's high validation accuracy reflects label imbalance.
The local teacher often labels whole solutions as incorrect, limiting the value
of synthetic supervision. These failures are visible in retained predictions
and confusion matrices.

The public-score replay is the appropriate way to check original evaluation
numbers. See `experiments/official-replay/`; its results must not be attributed
to locally trained models.

The public replay matches PRM 78.2%, ORM 72.4%, and all four Appendix F PRM scoring variants. The normalized majority implementation produces 69.20% instead of the published 69.6%; raw-string voting gives 69.10%. The original voting implementation is absent, so neither is labeled an exact majority-baseline reproduction.
