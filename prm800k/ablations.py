"""Matched local teacher/answer supervision and acquisition experiments.

This is a small mechanism ablation, not the paper's large teacher experiment.
"""
from pathlib import Path

from .data import candidates, split_by_problem
from .evaluation import evaluate, group_samples
from .experiment import FIXTURES, environment
from .io import read_jsonl, sha256, write_json, write_jsonl
from .model import ModelConfig, load_model
from .scoring import score_candidates
from .supervision import active_select, synthetic_rows
from .training import train


def run_ablation(teacher_path, output, device='auto', seeds=(42,43,44), pool_limit=160):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Ablation output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    raw = list(read_jsonl(FIXTURES/'train.jsonl.gz'))
    # Same split as the checked-in teacher's demo run (seed 42).
    train_rows, validation_rows = split_by_problem(raw, seed=42)
    teacher, _ = load_model(teacher_path, device)
    teacher_digest = sha256(teacher_path)
    def label(rows, limit):
        pool = score_candidates(list(candidates(rows))[:limit], teacher)
        for sample in pool:
            sample['teacher'] = teacher_digest
        return pool
    pool = label(train_rows, pool_limit)
    validation_pool = label(validation_rows, max(24, pool_limit//4))
    labels = list(synthetic_rows(pool))
    validation = list(synthetic_rows(validation_pool))
    write_jsonl(output/'teacher_pool.jsonl.gz', pool)
    write_jsonl(output/'synthetic_train.jsonl.gz', labels)
    write_jsonl(output/'synthetic_validation.jsonl.gz', validation)
    testing = list(candidates(read_jsonl(FIXTURES/'test.jsonl.gz')))
    results = {}
    import random
    for seed in seeds:
        conditions = {'process': ('prm', labels), 'outcome_teacher': ('orm-teacher', labels),
                      'outcome_answer': ('orm', labels)}
        budget = max(8, len(pool)//2)
        active = active_select(pool, budget, per_problem=False)
        uniform = random.Random(seed).sample(pool, min(budget, len(pool)))
        conditions['active_half'] = ('prm', list(synthetic_rows(active)))
        conditions['uniform_half'] = ('prm', list(synthetic_rows(uniform)))
        for name, (kind, rows) in conditions.items():
            destination = output/f'{name}-seed{seed}'
            write_jsonl(destination/'training_labels.jsonl.gz', rows)
            model, metadata = train(rows, validation, destination, kind=kind, device=device, seed=seed,
                                    config=ModelConfig(context=256), max_examples=3000)
            if kind == 'prm':
                scored = score_candidates(testing, model)
                method = 'prm'
            else:
                from .scoring import predict
                probs = predict(model, [(s['problem'],s['steps']) for s in testing])
                scored = [dict(s, orm_score=p[1]) for s,p in zip(testing,probs)]
                method = 'orm'
            grouped = sorted(group_samples(scored).items())
            metrics = evaluate(grouped, [1,2,4], methods=[method], seed=seed)
            write_json(destination/'evaluation.json', metrics)
            results[f'{name}-seed{seed}'] = metrics['methods'][method]
    write_json(output/'ablation.json', {'conditions': results, 'seeds': list(seeds),
         'teacher_checkpoint_sha256': teacher_digest, 'pool_size': len(pool), 'threshold':0.2,
         'scope':'Local teacher, released active-biased candidate pool, not paper Figure 4/5 effect sizes',
         'acquisition':'global 80% convincing wrong + 20% remaining; compared with uniform subset of same size',
         'environment': environment(str(next(teacher.parameters()).device))})
    lines = ['# Local mechanism ablations', '', '| Condition | N=1 | N=2 | N=4 |', '|---|---:|---:|---:|']
    for name, values in results.items():
        lines.append('| '+name+' | '+' | '.join(f'{x:.2%}' for x in values['exact_mean'])+' |')
    lines.extend(['', 'These are local mechanism tests with a small, weak teacher and an actively selected',
                  'candidate pool. They do not establish the paper’s supervision or data-efficiency gains.',
                  'Matched supervision uses identical candidates across the three supervision types.',
                  'The smaller active and uniform conditions use equal budgets. Test problems are excluded.'])
    (output/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    return results
