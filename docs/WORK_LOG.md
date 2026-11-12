# Implementation record

Baseline: upstream `openai/prm800k` commit `7ecc794703b2877f63226f2477a49b34f9b25163`.
Host: Apple M1 Pro, 16 GiB unified memory, macOS 26.6.1, arm64.

The original release contains human annotations (LFS pointers in this checkout),
MATH splits, labeling instructions, an answer grader and a remote-only evaluator.
It contains no training code or generator/checkpoint. The paper's GPT-4 base,
MathMix corpus, teacher and full ORM training corpus were not released here.

Implementation priorities:

1. Preserve original assets and grading semantics; fix packaging/imports.
2. Stream and verify released data; bundle a small, traceable offline subset.
3. Implement causal process/outcome training, local generation, scoring,
   synthetic supervision and active sample acquisition.
4. Preserve best-of-N sampling with missing generations counted as failures;
   add exact expectation and bounded-memory public-score evaluation.
5. Run actual M1 experiments and regression tests; retain inputs, outputs,
   environment, checkpoints and analysis in the repository.
6. Publish English/Chinese documentation, diagrams and an actual recorded demo.

No claim of full paper-scale reproduction is made without released artifacts
and matching experimental evidence. See `REPRODUCTION.md` for the final audit.

Completed method work includes causal PRM/ORM, local Qwen generation and LoRA,
synthetic teacher targets, active acquisition, matched three-seed ablations,
exact/Monte Carlo evaluation, subject/level/difficulty breakdowns and HTML reports.

Actual validation found a phase-1 schema discrepancy: human replacements may be
completion objects, not strings. A regression test now covers the real form.
A first parallel score-download attempt exposed a temporary-name collision.
Unique range filenames and an exclusive download lock corrected it; corrupted
parts were discarded, then the complete asset passed Azure Content-MD5 verification.

Recorded negative outcomes are preserved: short-budget generation was truncated,
longer stochastic generation did not solve the two selected MATH problems, and
the tiny PRM did not exceed the local ORM. CPU/MPS Qwen parity was independently
checked to distinguish sampling/model limitations from a device computation bug.

Annotation-protocol follow-up adds an explicit phase-1/phase-2 review state
machine and frozen-example trainer. All 952 bundled official trajectories replay
successfully. Real records exposed two details now covered by regression tests:
an unselected human candidate may coexist with a selected model candidate, and a
terminal alternative may repeat the original step text with a different rating.
Chosen indices and original completion identity, rather than text deduplication,
determine context and targets.

The new recorded experiment runs 18 MPS models across three paired comparisons
and three seeds, plus four CPU smoke models. The comparisons equalize per-problem
target quotas, initialization and optimizer steps. No human labels are invented
for the 5,001 steps after first-error stops. A separately identified weak-teacher
family tests synthetic full-trajectory versus first-error supervision. Frozen
examples, state traces, source hashes, checkpoints, predictions and bilingual
analysis are retained. Negative selection outcomes are reported without claims
of general inferiority or statistical significance.

The full-recording follow-up materializes all four official annotation files as
101,599 individual JSON objects, retaining raw JSONL and per-record hashes. The
full local run uses 619,636 training and 158,137 validation step labels without a
reservoir cap, trains the width-32/context-128 PRM for one epoch on MPS, reloads
the persisted checkpoint, and scores 2,664 complete test candidates across
457 problems (33,864 step predictions). Every one of 9,682 updates retains
weights, AdamW moments, raw/clipped gradients and loss-bearing outputs.

The previous 18 protocol conditions were rerun with the same complete tracing;
their exact best-of-N results match the previous runs. The verification tools
compare every extracted JSON object with its official source, reconstruct
optimizer transitions, verify sample/epoch coverage, recompute inference on CPU
and replay evaluation selections. Full multi-GB traces and raw expanded data
stay in the local project; source/record indexes, manifests, final models,
every-update CSV, figures, analysis, verification and previews are versioned.

The full-data model remains weak: validation accuracy 64.67% is close to the
63.91% majority baseline, macro-F1 is 0.311, and best-of-4 on the fixed
451-problem cohort is 22.96%, below its exact N=1 rate of 24.11%. Recording a
complete local training run does not establish paper-scale model capability.
