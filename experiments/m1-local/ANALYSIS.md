# PRM800K · Local experiment

32 problems; 100 subsampling trials; seed 42.

| Method | N=1 | N=2 | N=4 |
|---|---:|---:|---:|
| prm | 29.98% | 29.63% | 25.42% |
| orm | 29.98% | 30.68% | 28.66% |
| majority | 31.62% | 31.62% | 35.88% |
| random | 31.62% | 31.62% | 31.62% |
| oracle | 31.62% | 51.41% | 71.09% |
| prm_minimum | 29.98% | 26.88% | 19.38% |
| prm_positive_only | 29.98% | 29.58% | 24.17% |
| prm_positive_minimum | 29.98% | 26.67% | 22.29% |

Exact expectation is reported for score-based selection when present. Majority,
random and oracle use seeded sampling. Missing/answerless candidates count as
failed generation slots. Standard deviation describes subsampling variability,
not uncertainty over the entire population of MATH problems.

The bundled local experiment uses small models initialized from scratch and
an active-learning-biased subset of the released data. Its PRM/ORM comparison
is a functional demonstration, not evidence for the paper's effect size.
Short context truncation, class imbalance and the small training set limit
quality. The report shows every retained test candidate; no example is chosen
for model success. Validation problems and test problems are excluded from training.

See `prm_training.json`, `orm_training.json`, `environment.json`, `inputs.json`,
`scored.jsonl.gz` and checkpoint files for the corresponding run when available.

## Classification diagnostics

PRM: accuracy 53.84%; majority-class baseline 56.65%; balanced accuracy 34.84%; macro-F1 0.315. Training context truncated for 5195/6000 examples.

ORM: accuracy 93.02%; majority-class baseline 93.02%; balanced accuracy 50.00%; macro-F1 0.482. Training context truncated for 505/511 examples.

