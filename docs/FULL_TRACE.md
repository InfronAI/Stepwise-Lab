# Full local data, training and inference records

The `full-local` workflow uses the complete released phase-1 and phase-2 training
files. It applies the existing usability and step-label filters, assigns problems
to an 80%/20% train/validation split, and trains on **every resulting training
example** without a reservoir limit. All phase-2 test candidates with complete
pregenerated steps and an answer are scored. The PRM is trained and reloaded on
the local machine; no inference API is used.

This is a full-data execution of the project's small causal byte transformer.
The default is one epoch, width 32, two layers, context 128 and batch size 64 to
keep the complete trace practical on M1. It is not the paper's private GPT-4 PRM
and does not reproduce its capacity, training budget or quality.

## Run

```bash
# Four immutable official files, verified against upstream LFS SHA-256.
prm800k materialize-data --output datasets/prm800k

# Every eligible training label; all updates, validation and inference recorded.
prm800k full-local --dataset datasets/prm800k --device mps \
  --epochs 1 --context 128 --width 32 --batch-size 64 \
  --output experiments/runs/full-local

# Verify raw records, source snapshot, checksums, every optimizer transition,
# validation coverage, every test prefix and all evaluation selections.
python scripts/validate_full_trace.py --dataset datasets/prm800k \
  --run experiments/runs/full-local
python scripts/summarize_full_trace.py experiments/runs/full-local
```

Use a new output directory. `--device cpu` executes the same workflow. Full
recording writes a checkpoint for **every** optimizer update and consumes several
GB of disk; it also makes execution slower than the small fixture demo. There is
no automatic cleanup or rotation of intermediate states.

## Local raw data

| Asset | Original JSONL records | Individual JSON files |
|---|---:|---|
| Phase 1 train | 949 | `datasets/prm800k/records/phase1_train/` |
| Phase 1 test | 106 | `datasets/prm800k/records/phase1_test/` |
| Phase 2 train | 97,782 | `datasets/prm800k/records/phase2_train/` |
| Phase 2 test | 2,762 | `datasets/prm800k/records/phase2_test/` |

All **101,599** records are retained, including records that are unsuitable for
training. Original `.jsonl` files live under `datasets/prm800k/raw/`; each line is
parsed and written as a separate JSON object under a shard folder containing at
most 1,000 records. The source files are JSONL, so this is record extraction, not
an alteration of the annotation format. Record indexes link sequence numbers,
relative paths, problem IDs, usability and individual file hashes. Extraction
preserves the complete JSON object, including labeler metadata, candidates and
unlabeled tail text.

The raw data and expanded records are local files inside the project, excluded
from Git to avoid duplicating the official large-data hosting. The download
recipe, source commit, source hashes, record indexes and verification receipt
are versioned. Running `materialize-data` recreates the local layout.

## What is saved at each stage

Paths below are relative to `experiments/full-data-local/`, the recorded run.

| Stage | Saved evidence |
|---|---|
| Source and configuration | `config.json`, `inputs.json`, source hashes and complete `source_snapshot/`; dataset provenance |
| Workflow state | `pipeline.json`: stage start/end timestamps, complete/failed status and failure reason |
| Projection and encoding | `details/training/train_examples.jsonl.gz` and `validation_examples.jsonl.gz`: source trajectory, selected prefix, target step/rating, byte-token IDs, solution offset and truncation |
| Selected training set | `details/training/selected_examples.json.gz`: exact IDs used by training and validation |
| Initialization | `details/training/states/update_000000.pt.gz`: initial weights, optimizer configuration and random states |
| Every training batch | `details/training/batches/update_*.json.gz`: ordered example IDs, loss, loss-bearing logits/targets, endpoint probabilities and gradient norm |
| Every parameter update | `details/training/states/update_*.pt.gz`: updated weights, AdamW moments, raw and clipped gradients, update/epoch number, random states and corresponding batch record |
| Every validation example | `details/training/validation/epoch_*.jsonl.gz`: ordered example IDs, logits, probabilities, targets and losses; epoch files store confusion matrices and aggregate metrics |
| Final local model | `prm.pt`, `prm_training.json`; the inference stage explicitly reloads `prm.pt` |
| Inference input and answer grading | `test_candidates.jsonl.gz`: complete original candidate, answer, reference answer and grader decision |
| Every reasoning-step prediction | `details/inference/candidates/*.json.gz`: each causal prefix, exact tokens, truncation, endpoint logits/probabilities and cumulative log score |
| Candidate reduction and ranking | `scored.jsonl.gz`, `ranked_selections.jsonl.gz`: four score reductions, candidate order and top-score ties |
| Every evaluation draw | `details/evaluation_*/problems/*.json.gz`: sampled candidate indices, winner and correctness for every trial/N; score-tier contributions to exact expectations |
| Final analysis and integrity | Evaluation JSON, summary, figures, analysis, validation receipt and SHA-256 manifests |

PRM loss uses one endpoint per example. ORM tracing, available through `train
--trace`, records every supervised solution-token logit and target. Unused hidden
layer activations and GPU implementation temporaries are not dumped: the trace
records the algorithm's inputs, supervised outputs, gradients and states, with
enough information to replay each optimizer update and inference prefix.

CPU-recomputed token cross-entropies are labeled as such; the actual device batch
loss is also preserved. Raw and clipped gradients are separate. Clipping uses a
global norm of 1.0. Tensor archives contain CPU tensors and can be read safely
with `load_tensor_archive`, which uses `torch.load(weights_only=True)`:

```python
from prm800k.tracing import load_tensor_archive

state = load_tensor_archive(
    "experiments/full-data-local/details/training/states/update_000001.pt.gz"
)
print(state["batch"]["example_ids"], state["batch"]["batch_loss"])
print(state["optimizer_state_dict"]["param_groups"])
```

## Evaluation coverage and interpretation

Every usable full phase-2 test candidate is scored along its entire generated
trajectory, including the human-unlabeled suffix. Those suffixes receive model
predictions, **not invented human training targets**. Phase-1 test annotations
are downloaded and retained; they are not treated as complete pregenerated
candidate solutions.

All-test N=1 is reported on every eligible test problem. Best-of-1/2/4 is also
reported on a **fixed cohort with at least four labeled candidates**. The full
test set has variable candidate counts, so absent labeled candidates are not
silently fabricated as failed generations. Ranking across all available
candidates is reported separately, with uniform expected correctness across
exact score ties. Comparisons with the 32-problem fixture or the original paper
must account for the different data, context and model sizes.

## Trace existing commands

```bash
prm800k annotation-ablate --device mps --seeds 42,43,44 --budget 1024 \
  --teacher experiments/m1-local/prm.pt --trace \
  --output experiments/runs/annotation-traced
prm800k train --train prm800k/fixtures/train.jsonl.gz --kind prm --trace \
  --output experiments/runs/training-traced
prm800k score --input experiments/m1-generation/generated.jsonl.gz \
  --prm experiments/full-data-local/prm.pt --trace \
  --output experiments/runs/scored.jsonl.gz
prm800k evaluate --input experiments/runs/scored.jsonl.gz --method prm --ns 1,2,4 \
  --trace --output experiments/runs/evaluation-traced
```

Historical experiments remain unchanged. Missing historical intermediate tensors
cannot be recovered from final checkpoints; traced runs are actual new executions.
Bulky step archives for the recorded full-data and protocol reruns remain locally
under `details/` or `trace/`. Their manifests, final weights, summary metrics and
representative previews are versioned. The run source snapshot protects the
full-data evidence against later source edits.
