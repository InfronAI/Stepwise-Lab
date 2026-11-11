# PRM800K · Candidate evaluation

2 problems; 100 subsampling trials; seed 42.

| Method | N=1 | N=2 | N=4 |
|---|---:|---:|---:|
| prm | 0.00% | 0.00% | 0.00% |
| orm | 0.00% | 0.00% | 0.00% |
| majority | 0.00% | 0.00% | 0.00% |

Exact expectation is reported for score-based selection when present. Majority,
random and oracle use seeded sampling. Missing/answerless candidates count as
failed generation slots. Standard deviation describes subsampling variability,
not uncertainty over the entire population of MATH problems.

This run evaluates supplied candidate scores. It does not train reward models.
Check the adjacent input and source manifests for dataset coverage, generator
revision, candidate counts, missing answers and applicable limitations.

Eight fresh attempts on two official MATH problems produced five extracted answers and zero correct answers. One reached the token limit. All failed candidates are retained. This is a small execution check, not a benchmark result.
