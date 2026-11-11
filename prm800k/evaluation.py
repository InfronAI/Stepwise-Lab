"""Exact and Monte Carlo best-of-N, with one-problem-at-a-time SQLite storage."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from .grading.math_normalize import normalize_answer
from .io import read_jsonl, sha256, write_json

METHODS = {'prm': 'prm_score', 'orm': 'orm_score', 'prm_minimum': 'prm_minimum',
           'prm_positive_only': 'prm_positive_only', 'prm_positive_minimum': 'prm_positive_minimum'}


def answer_of(sample):
    return sample.get('answer', sample.get('given_answer'))


def group_samples(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['problem']].append(sample)
    return grouped


def validate_group(samples, population, ns):
    if population < len(samples) or population < 1:
        raise ValueError('population must be positive and >= number of stored candidates')
    if not ns or any(n < 1 or n > population for n in ns):
        raise ValueError('Each N must be in [1, population]')
    for sample in samples:
        if not isinstance(sample.get('is_correct'), bool):
            raise ValueError('is_correct must be boolean')


def probability_avoid(population, excluded, n):
    """Probability a size-n subset avoids `excluded` items, stably in log-space."""
    available = population-excluded
    if available < n:
        return 0.0
    if excluded == 0:
        return 1.0
    # Product over the shorter of excluded or n avoids large lgamma cancellation.
    if excluded < n:
        return math.exp(sum(math.log1p(-n/(population-i)) for i in range(excluded)))
    return math.exp(sum(math.log1p(-excluded/(population-i)) for i in range(n)))


def exact_problem(samples, ns, population, method):
    validate_group(samples, population, ns)
    key = METHODS[method]
    usable = [s for s in samples if answer_of(s) is not None]
    tiers = defaultdict(list)
    for sample in usable:
        score = sample[key]
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError(f'Non-finite or invalid {key}')
        tiers[score].append(int(sample['is_correct']))
    scores = sorted(tiers, reverse=True)
    result = np.zeros(len(ns))
    avoid = {}
    for n in ns:
        ratios = np.maximum(0.0, 1.0 - n / np.arange(population, 0, -1, dtype=float))
        avoid[n] = np.concatenate(([1.0], np.cumprod(ratios)))
    above = 0
    # The selected tier has >=1 sampled item, and no sampled higher-scored item.
    # Equal-score tie-breaking is uniform, as with the upstream random shuffle.
    for score in scores:
        group = tiers[score]
        fraction = sum(group)/len(group)
        for j, n in enumerate(ns):
            result[j] += (avoid[n][above]-avoid[n][above+len(group)])*fraction
        above += len(group)
    return result


def monte_carlo_problem(samples, ns, population, methods, trials, rng):
    validate_group(samples, population, ns)
    if trials < 1:
        raise ValueError('trials must be positive')
    valid = np.zeros(population, dtype=bool)
    correct = np.zeros(population, dtype=bool)
    answers = [None] * population
    for i, sample in enumerate(samples):
        valid[i] = answer_of(sample) is not None
        correct[i] = sample['is_correct']
        answers[i] = normalize_answer(answer_of(sample))
    order = np.stack([rng.permutation(population)[:max(ns)] for _ in range(trials)])
    result = {}
    for method in methods:
        outcomes = np.zeros((trials, len(ns)), dtype=np.float64)
        if method in METHODS:
            values = np.full(population, -np.inf)
            for i, sample in enumerate(samples):
                if valid[i]:
                    value = sample[METHODS[method]]
                    if not math.isfinite(value):
                        raise ValueError(f'Non-finite score for {method}')
                    values[i] = value
            draws = values[order]
            for j, n in enumerate(ns):
                winner = order[np.arange(trials), draws[:, :n].argmax(axis=1)]
                outcomes[:, j] = valid[winner] & correct[winner]
        elif method in ('majority', 'random', 'oracle'):
            for trial in range(trials):
                for j, n in enumerate(ns):
                    indices = [int(i) for i in order[trial, :n] if valid[i]]
                    if not indices:
                        continue
                    if method == 'oracle':
                        outcomes[trial, j] = any(correct[i] for i in indices)
                    elif method == 'random':
                        outcomes[trial, j] = correct[indices[0]]
                    else:
                        # Insertion order after shuffle gives random ties, no correctness-based tie breaking.
                        counts = Counter(answers[i] for i in indices)
                        answer = max(counts, key=counts.get)
                        winner = next(i for i in indices if answers[i] == answer)
                        outcomes[trial, j] = correct[winner]
        else:
            raise ValueError(f'Unknown method: {method}')
        result[method] = outcomes
    return result


def evaluate(groups, ns=(1, 2, 4, 8), *, population=None, methods=('prm', 'orm', 'majority'), trials=100, seed=42, exact=True):
    """`groups` yields (problem, samples). Missing slots always remain in the population."""
    ns = sorted(set(ns))
    if not ns or any(n < 1 for n in ns) or trials < 1 or not methods:
        raise ValueError('positive Ns/trials and at least one method required')
    methods = list(dict.fromkeys(methods))
    if any(m not in {*METHODS, 'majority', 'random', 'oracle'} for m in methods):
        raise ValueError('Unknown evaluation method')
    rng = np.random.default_rng(seed)
    trial_totals = {m: np.zeros((trials, len(ns))) for m in methods}
    exact_totals = {m: np.zeros(len(ns)) for m in methods if m in METHODS and exact}
    per_problem = []
    total = 0
    count = 0
    start = time.perf_counter()
    for problem, samples in groups:
        count += 1
        p = population or len(samples)
        total += p
        simulation = monte_carlo_problem(samples, ns, p, methods, trials, rng)
        record = {'problem': problem, 'stored_samples': len(samples), 'population': p,
                  'answer_correct_samples': sum(s['is_correct'] and answer_of(s) is not None for s in samples),
                  'subject': samples[0].get('subject') if samples else None,
                  'level': samples[0].get('level') if samples else None}
        for method in methods:
            trial_totals[method] += simulation[method]
            record[method] = simulation[method].mean(0).tolist()
            if method in exact_totals:
                values = exact_problem(samples, ns, p, method)
                exact_totals[method] += values
                record[method] = values.tolist()
        per_problem.append(record)
    if not count:
        raise ValueError('No problems to evaluate')
    metrics = {}
    for method in methods:
        rates = trial_totals[method]/count
        metrics[method] = {'mean': rates.mean(0).tolist(), 'std': rates.std(0).tolist(),
                           'monte_carlo_standard_error': (rates.std(0)/math.sqrt(trials)).tolist()}
        if method in exact_totals:
            metrics[method]['exact_mean'] = (exact_totals[method]/count).tolist()
    breakdowns = {}
    for field in ('subject', 'level'):
        buckets = defaultdict(list)
        for record in per_problem:
            if record[field] is not None:
                buckets[str(record[field])].append(record)
        breakdowns[field] = {key: {'problems': len(rows), 'methods': {m: np.mean([r[m] for r in rows], axis=0).tolist() for m in methods}} for key, rows in buckets.items()}
    ordered = sorted(per_problem, key=lambda r: (r['answer_correct_samples']/r['population'], r['problem']))
    quintiles = np.array_split(np.arange(count), min(5, count))
    breakdowns['generator_difficulty'] = {f'quintile_{i+1}_hardest_first': {'problems': len(indices), 'methods': {m: np.mean([ordered[j][m] for j in indices], axis=0).tolist() for m in methods}} for i, indices in enumerate(quintiles)}
    return {'ns': ns, 'methods': metrics, 'breakdowns': breakdowns, 'problems': count, 'population_slots': total,
            'trials': trials, 'seed': seed, 'seconds': time.perf_counter()-start,
            'per_problem': per_problem,
            'protocol': 'without replacement; missing/answerless generations fail; shared draws across methods and N; uniform score ties',
            'uncertainty': 'std is subsampling variability, not a confidence interval over the population of math problems'}


def build_index(source, database, limit=None):
    """Stream a multi-GB release into a compact disk-backed index, atomically."""
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    temp = database.with_name(database.name + '.partial')
    if temp.exists():
        temp.unlink()
    count = 0
    with sqlite3.connect(temp) as conn:
        conn.execute('CREATE TABLE problems (id INTEGER PRIMARY KEY, problem TEXT UNIQUE, subject TEXT, level INTEGER)')
        conn.execute('CREATE TABLE samples (problem INTEGER, answer TEXT, correct INTEGER, prm REAL, orm REAL)')
        problem_ids = {}
        buffer = []
        for row in read_jsonl(source):
            if limit is not None and count >= limit:
                break
            if not isinstance(row.get('is_correct'), bool):
                raise ValueError('is_correct must be boolean')
            for key in ('prm_score', 'orm_score'):
                if not math.isfinite(row[key]):
                    raise ValueError(f'Invalid {key}')
            if row['problem'] not in problem_ids:
                problem_ids[row['problem']] = len(problem_ids)
                conn.execute('INSERT INTO problems VALUES (?,?,?,?)', (problem_ids[row['problem']], row['problem'], row.get('subject'), row.get('level')))
            buffer.append((problem_ids[row['problem']], answer_of(row), int(row['is_correct']), row['prm_score'], row['orm_score']))
            count += 1
            if len(buffer) == 5000:
                conn.executemany('INSERT INTO samples VALUES (?,?,?,?,?)', buffer)
                buffer.clear()
                if count % 100000 == 0:
                    print(f'Indexed {count:,} samples', flush=True)
        conn.executemany('INSERT INTO samples VALUES (?,?,?,?,?)', buffer)
        conn.execute('CREATE INDEX by_problem ON samples(problem)')
        conn.commit()
    temp.replace(database)
    source_hash = sha256(source) if Path(str(source)).is_file() else None
    write_json(database.with_suffix('.source.json'), {'source': str(source), 'source_sha256': source_hash, 'rows': count, 'row_limit': limit, 'complete': limit is None})
    return count


def indexed_groups(database):
    with sqlite3.connect(database) as conn:
        # Each query returns only one problem; all solution text is omitted at import.
        for problem_index, problem, subject, level in conn.execute('SELECT id,problem,subject,level FROM problems ORDER BY problem'):
            samples = [{'problem': problem, 'answer': a, 'is_correct': bool(c), 'prm_score': p, 'orm_score': o}
                       for a, c, p, o in conn.execute('SELECT answer,correct,prm,orm FROM samples WHERE problem=? ORDER BY rowid', (problem_index,))]
            if subject is not None or level is not None:
                for sample in samples:
                    sample.update(subject=subject, level=level)
            yield problem, samples
