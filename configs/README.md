# Reproducible recipes

Run configurations are captured in each experiment's `config.json`, training
metrics and generation manifests. The CLI is the source of executable defaults.

| Recipe | Command |
|---|---|
| Offline M1 demo | `prm800k demo --device mps --seed 42 --epochs 2 --max-examples 6000 --context 256 --output experiments/runs/demo` |
| CPU smoke run | `prm800k demo --device cpu --epochs 1 --max-examples 256 --context 128 --output experiments/runs/cpu` |
| Matched ablations | `prm800k ablate --teacher experiments/m1-local/prm.pt --device mps --seeds 42,43,44 --pool-limit 96 --output experiments/runs/ablations` |
| Official replay | `prm800k evaluate --database .cache/scored.sqlite --population 1860 --ns 1,10,25,50,100,200,500,1000,1860 --method prm,orm --trials 400 --output experiments/runs/official` |

Use a new output directory per experiment. The default demo intentionally keeps
models and contexts small. To scale up, increase training context and example
budget explicitly; do not interpret a memory-constrained demo as the paper's
pretrained model architecture or training recipe.
