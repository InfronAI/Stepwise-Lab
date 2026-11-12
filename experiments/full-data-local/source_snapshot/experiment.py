"""Offline, auditable end-to-end experiment."""
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
import subprocess
import time

from .data import candidates, split_by_problem, audit
from .evaluation import evaluate, group_samples
from .io import read_jsonl, sha256, write_json, write_jsonl, problem_id
from .report import render_report

FIXTURES = Path(__file__).resolve().parent/'fixtures'


def environment(device=None):
    packages = {}
    for name in ['prm800k', 'numpy', 'torch', 'sympy', 'matplotlib', 'transformers']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
            'packages': packages, 'device': device, 'base_git_commit': commit}


def run_demo(output='experiments/runs/demo', device='auto', seed=42, epochs=2, max_examples=6000, context=256):
    from .model import ModelConfig
    from .training import train
    from .scoring import score_candidates
    from .supervision import active_select, synthetic_rows

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; choose a new --output to preserve experiment evidence')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_path, test_path = FIXTURES/'train.jsonl.gz', FIXTURES/'test.jsonl.gz'
    print('1/6 Read official-data fixtures and split by problem', flush=True)
    rows = list(read_jsonl(train_path))
    test_rows = list(read_jsonl(test_path))
    train_rows, validation = split_by_problem(rows, seed=seed)
    training_problems = {r['question']['problem'] for r in rows}
    test_problems = {r['question']['problem'] for r in test_rows}
    if training_problems & test_problems:
        raise ValueError('Train/test problem leakage')
    inputs = {'train': {'file': str(train_path.name), 'sha256': sha256(train_path)},
              'test': {'file': str(test_path.name), 'sha256': sha256(test_path)},
              'train_problem_ids': sorted({problem_id(r['question']['problem']) for r in train_rows}),
              'validation_problem_ids': sorted({problem_id(r['question']['problem']) for r in validation}),
              'test_problem_ids': sorted(problem_id(p) for p in test_problems), 'overlap': 0}
    write_json(output/'inputs.json', inputs)
    config = ModelConfig(context=context)
    write_json(output/'config.json', {'seed': seed, 'epochs': epochs, 'max_examples': max_examples, 'model': asdict(config)})
    print('2/6 Train PRM on positive / neutral / negative step labels', flush=True)
    prm, prm_meta = train(train_rows, validation, output, device=device, seed=seed, epochs=epochs,
                           config=config, max_examples=max_examples)
    print('3/6 Train ORM on final-answer labels at every solution token', flush=True)
    orm, orm_meta = train(train_rows, validation, output, kind='orm', device=device, seed=seed,
                           config=config, max_examples=max_examples)
    print('4/6 Score complete held-out solutions', flush=True)
    scored = score_candidates(list(candidates(test_rows)), prm, orm)
    write_jsonl(output/'scored.jsonl.gz', scored)
    print('5/6 Evaluate best-of-N and all four PRM scoring policies', flush=True)
    grouped = group_samples(scored)
    min_candidates = min(map(len, grouped.values()))
    ns = [n for n in (1, 2, 4, 8, 16) if n <= min_candidates]
    evaluation = evaluate(sorted(grouped.items()), ns, seed=seed, methods=['prm', 'orm', 'majority', 'random', 'oracle', 'prm_minimum', 'prm_positive_only', 'prm_positive_minimum'])
    print('6/6 Acquire training candidates, export synthetic labels and report', flush=True)
    # Acquisition only sees training data, never the held-out evaluation set.
    training_pool = list(candidates(train_rows))[:48]
    pool = score_candidates(training_pool, prm)
    selected = active_select(pool, 16, per_problem=False)
    write_jsonl(output/'active_selected.jsonl.gz', selected)
    write_jsonl(output/'synthetic_labels.jsonl.gz', synthetic_rows(pool))
    write_json(output/'data_audit.json', audit([train_path, test_path]))
    env = environment(prm_meta['device'])
    env['total_seconds'] = time.perf_counter()-started
    write_json(output/'environment.json', env)
    report = render_report(output, evaluation, scored, {'prm': prm_meta, 'orm': orm_meta}, env)
    write_json(output/'artifacts.json', {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(f'Completed in {env["total_seconds"]:.1f}s. Report: {report.resolve()}', flush=True)
    return report
