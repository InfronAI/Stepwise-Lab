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
