# Recorded experiments

These directories contain real executions on the available Apple M1 Pro (16 GiB).
No experiment claims a full reproduction of the private GPT-4 training pipeline.

| Directory | Purpose |
|---|---|
| `m1-local` | Offline PRM/ORM training and selection over 32 held-out problems, 184 candidates |
| `cpu-smoke` | Complete CPU fallback with smaller training/context budgets |
| `m1-ablations` | Three seeds × five matched supervision/acquisition conditions |
| `annotation-protocols` | Three paired annotation-policy comparisons × three seeds; 18 MPS runs, official-record state-machine replay and synthetic horizon diagnostics |
| `annotation-cpu-smoke` | Four CPU runs verifying the two human-label protocol comparisons with smaller budgets/context |
| `annotation-protocols-traced` | Actual rerun of all 18 annotation-protocol models with every update, validation prediction, inference prefix and selection draw recorded |
| `full-data-local` | Uncapped training on usable official train labels, individual-JSON source records, local checkpoint reload and all complete phase-2 test candidates; every intermediate retained |
| `m1-generation-192tokens` | Eight fresh attempts on two official MATH test problems; all truncated without final answers |
| `m1-generation` | Same problems at 768-token budget; five extracted answers, no correct answers |
| `m1-sft` | Actual four-example generator LoRA SFT and two-example text-adaptation execution |
| `m1-adapter-generation` | Load the saved SFT adapter and sample an explicitly handwritten integration prompt |
| `m1-arithmetic` | Lower-temperature sampling on the same handwritten prompt; still not a benchmark |
| `m1-greedy` | Deterministic generation diagnostic, separate from stochastic best-of-N |
| `official-replay` | Evaluate released PRM/ORM scores; no local training of original models |

`generator-device-parity.json` compares a Qwen CPU/MPS forward pass and greedy
continuation: maximum final-token logit difference 0.000122, matching continuations
and agreement at atol/rtol 0.001. Both correctly answer the simple multiplication
prompt in that diagnostic. This distinguishes generator sampling/prompt sensitivity
from a Metal computation mismatch.

The recorded sampled-generation failures are retained. A weak model, stochastic
sampling, short budgets and strict answer extraction can all limit candidate
quality. A verifier cannot select a correct answer when none was generated.
These tiny measurements should not be generalized to the model's overall ability.

The local PRM's best-of-4 rate is below the ORM and majority baseline. ORM's
93.02% validation accuracy comes from always choosing the dominant negative
class; its balanced accuracy is 50%. Read confusion matrices and class-aware
metrics alongside the selection curves. The local teacher is weak and all
synthetic supervision results must be interpreted accordingly.

Large public downloads and SQLite caches are excluded from Git. Compact input
fixtures, generated candidates, checkpoints/adapters, source manifests and
analysis are retained here. `artifacts.json` checksums each run's files;
`validation.json` records final artifact checks. `runs/` is ignored scratch space
for new experiments and failed/debug attempts.
