"""Lossless annotation interpretation and leakage-resistant problem splits."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

from .io import problem_id, read_jsonl


@dataclass
class StepExample:
    problem: str
    prefix: list[str]
    step: str
    rating: int
    source: str


def usable(row):
    return not (row.get('is_quality_control_question') or row.get('is_initial_screening_question')) and row['label']['finish_reason'] in ('solution', 'found_error')


def step_examples(rows):
    """Each alternative sees only the chosen earlier trajectory, never its siblings.

    Human replacements are positive. Null/unrated/flagged labels are skipped.
    Never continue through an unchosen or incorrect branch.
    """
    for row in rows:
        if not usable(row):
            continue
        problem = row['question']['problem']
        prefix = []
        for index, step in enumerate(row['label']['steps']):
            for alternative in step['completions']:
                if alternative.get('rating') in (-1, 0, 1) and not alternative.get('flagged'):
                    yield StepExample(problem, prefix.copy(), alternative['text'], alternative['rating'], f'{problem_id(problem)}:{index}')
            chosen = step.get('chosen_completion')
            human = human_text(step.get('human_completion'))
            if chosen is None:
                if human:
                    yield StepExample(problem, prefix.copy(), human, 1, f'{problem_id(problem)}:{index}:human')
                    prefix.append(human)
                else:
                    break
            else:
                if not isinstance(chosen, int) or not 0 <= chosen < len(step['completions']):
                    raise ValueError('chosen_completion index out of bounds')
                completion = step['completions'][chosen]
                if completion.get('rating') not in (0, 1) or completion.get('flagged'):
                    break
                prefix.append(completion['text'])


def human_text(completion):
    # Phase 1 actually contains completion objects, despite the README shorthand.
    if isinstance(completion, dict):
        return None if completion.get('flagged') else completion.get('text')
    if completion is None or isinstance(completion, str):
        return completion
    raise ValueError('human_completion must be a text, object or null')


def chosen_steps(row):
    result = []
    for step in row['label']['steps']:
        chosen = step.get('chosen_completion')
        if chosen is not None:
            result.append(step['completions'][chosen]['text'])
        elif human_text(step.get('human_completion')):
            result.append(human_text(step['human_completion']))
        else:
            break
    return result


def candidates(rows):
    """Use complete pregenerated solutions; never mark a prefix as a full solution."""
    from .grading.grader import grade_answer

    for index, row in enumerate(rows):
        if not usable(row):
            continue
        q = row['question']
        steps = q.get('pre_generated_steps')
        answer = q.get('pre_generated_answer')
        if not steps or answer is None:
            continue
        yield {
            'id': f'{problem_id(q["problem"])}:{index}', 'problem': q['problem'],
            'steps': steps, 'answer': answer, 'ground_truth_answer': q['ground_truth_answer'],
            'is_correct': grade_answer(answer, q['ground_truth_answer']),
            'source': 'released PRM800K; active-learning-biased candidates',
        }


def split_by_problem(rows, fraction=0.2, seed=42):
    if not 0 < fraction < 1:
        raise ValueError('validation fraction must be between 0 and 1')
    problems = sorted({r['question']['problem'] for r in rows})
    if len(problems) < 2:
        raise ValueError('At least two distinct problems required for a split')
    random.Random(seed).shuffle(problems)
    validation = set(problems[:max(1, min(len(problems)-1, round(len(problems)*fraction)))])
    return ([r for r in rows if r['question']['problem'] not in validation],
            [r for r in rows if r['question']['problem'] in validation])


def audit(paths):
    summary = {}
    for path in paths:
        counts = Counter()
        problems = set()
        for row in read_jsonl(path):
            counts['rows'] += 1
            counts['usable_rows'] += int(usable(row))
            counts[row['label']['finish_reason']] += 1
            problems.add(problem_id(row['question']['problem']))
            for example in step_examples([row]):
                counts[f'rating_{example.rating}'] += 1
        summary[str(path)] = dict(counts, problems=len(problems))
    return summary


class AnnotationSource:
    """Re-iterable streaming source with a stable hash split for full-size training."""
    def __init__(self, paths, partition=None, seed=42, fraction=0.2):
        self.paths = list(paths)
        self.partition = partition
        self.seed = seed
        self.fraction = fraction

    def __iter__(self):
        import hashlib
        for path in self.paths:
            for row in read_jsonl(path):
                if self.partition is not None:
                    value = int.from_bytes(hashlib.sha256(f'{self.seed}:{row["question"]["problem"]}'.encode()).digest()[:8], 'big') / 2**64
                    is_validation = value < self.fraction
                    if is_validation != (self.partition == 'validation'):
                        continue
                yield row
