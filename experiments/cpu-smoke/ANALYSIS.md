# PRM800K · Local experiment

32 problems; 100 subsampling trials; seed 42.

| Method | N=1 | N=2 | N=4 |
|---|---:|---:|---:|
| prm | 29.98% | 30.25% | 27.71% |
| orm | 29.98% | 31.79% | 27.56% |
| majority | 31.62% | 31.62% | 35.88% |
| random | 31.62% | 31.62% | 31.62% |
| oracle | 31.62% | 51.41% | 71.09% |
| prm_minimum | 29.98% | 31.71% | 31.25% |
| prm_positive_only | 29.98% | 30.46% | 28.54% |
| prm_positive_minimum | 29.98% | 31.74% | 32.02% |

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

PRM: accuracy 58.59%; majority-class baseline 58.98%; balanced accuracy 33.11%; macro-F1 0.246. Training context truncated for 245/256 examples.

ORM: accuracy 93.02%; majority-class baseline 93.02%; balanced accuracy 50.00%; macro-F1 0.482. Training context truncated for 256/256 examples.

