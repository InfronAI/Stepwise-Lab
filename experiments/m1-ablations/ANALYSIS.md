# Local mechanism ablations

| Condition | N=1 | N=2 | N=4 |
|---|---:|---:|---:|
| process-seed42 | 29.98% | 29.73% | 26.25% |
| outcome_teacher-seed42 | 29.98% | 34.91% | 37.14% |
| outcome_answer-seed42 | 29.98% | 33.87% | 33.81% |
| active_half-seed42 | 29.98% | 28.38% | 24.17% |
| uniform_half-seed42 | 29.98% | 29.00% | 25.00% |
| process-seed43 | 29.98% | 29.00% | 25.42% |
| outcome_teacher-seed43 | 29.98% | 30.64% | 34.64% |
| outcome_answer-seed43 | 29.98% | 30.64% | 34.64% |
| active_half-seed43 | 29.98% | 28.59% | 25.21% |
| uniform_half-seed43 | 29.98% | 30.15% | 26.46% |
| process-seed44 | 29.98% | 29.42% | 26.46% |
| outcome_teacher-seed44 | 29.98% | 28.33% | 23.33% |
| outcome_answer-seed44 | 29.98% | 27.92% | 23.12% |
| active_half-seed44 | 29.98% | 29.87% | 25.98% |
| uniform_half-seed44 | 29.98% | 30.57% | 26.04% |

These are local mechanism tests with a small, weak teacher and an actively selected
candidate pool. They do not establish the paper’s supervision or data-efficiency gains.
Matched supervision uses identical candidates across the three supervision types.
The smaller active and uniform conditions use equal budgets. Test problems are excluded.
