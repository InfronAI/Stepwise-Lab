# PRM800K · Precomputed-score evaluation

500 problems; 400 subsampling trials; seed 42.

| Method | N=1 | N=10 | N=25 | N=50 | N=100 | N=200 | N=500 | N=1000 | N=1860 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prm | 46.57% | 67.84% | 71.46% | 73.39% | 74.90% | 76.04% | 77.09% | 77.65% | 78.20% |
| orm | 46.57% | 66.93% | 69.64% | 70.82% | 71.52% | 71.92% | 72.29% | 72.50% | 72.40% |

Exact expectation is reported for score-based selection when present. Majority,
random and oracle use seeded sampling. Missing/answerless candidates count as
failed generation slots. Standard deviation describes subsampling variability,
not uncertainty over the entire population of MATH problems.

This run evaluates supplied candidate scores. It does not train reward models.
Check the adjacent input and source manifests for dataset coverage, generator
revision, candidate counts, missing answers and applicable limitations.

## Full-release validation

All 500 problems and 815,632 stored candidates were processed. The evaluation population retains 114,368 missing attempts.

| PRM scoring policy | Best-of-1860 |
|---|---:|
| product_neutral_positive | 78.20% |
| minimum_neutral_positive | 77.60% |
| product_neutral_negative | 77.40% |
| minimum_neutral_negative | 77.80% |

Normalized majority voting at the full population: **69.20%**.
The paper reports majority voting at 69.6%; this implementation gives about 69.2% with the released answer strings and upstream normalization. The original voting implementation was not released, so this baseline is not an exact reproduction.
This is score replay of the original released models, not training reproduction.
The source file passed byte-length and Azure Content-MD5 verification; its SHA-256 is in `source.json`.

### subject

| Group | Problems | PRM N=1860 | ORM N=1860 |
|---|---:|---:|---:|
| Algebra | 124 | 95.97% | 87.90% |
| Geometry | 41 | 56.10% | 53.66% |
| Prealgebra | 82 | 85.37% | 87.80% |
| Counting & Probability | 38 | 78.95% | 81.58% |
| Precalculus | 56 | 67.86% | 51.79% |
| Number Theory | 62 | 88.71% | 74.19% |
| Intermediate Algebra | 97 | 57.73% | 54.64% |

### level

| Group | Problems | PRM N=1860 | ORM N=1860 |
|---|---:|---:|---:|
| 1 | 43 | 95.35% | 90.70% |
| 5 | 134 | 56.72% | 46.27% |
| 4 | 128 | 75.78% | 75.00% |
| 3 | 105 | 89.52% | 80.95% |
| 2 | 90 | 92.22% | 88.89% |

### generator_difficulty

| Group | Problems | PRM N=1860 | ORM N=1860 |
|---|---:|---:|---:|
| quintile_1_hardest_first | 100 | 25.00% | 18.00% |
| quintile_2_hardest_first | 100 | 76.00% | 67.00% |
| quintile_3_hardest_first | 100 | 91.00% | 87.00% |
| quintile_4_hardest_first | 100 | 99.00% | 93.00% |
| quintile_5_hardest_first | 100 | 100.00% | 97.00% |
