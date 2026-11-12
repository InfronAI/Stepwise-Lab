"""Paper-style synthetic labels and convincing-wrong-answer acquisition."""
from collections import defaultdict
import math


def synthetic_rows(samples, threshold=0.2):
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1]')
    for sample in samples:
        probabilities = sample['step_probabilities']
        if len(probabilities) != len(sample['steps']) or not probabilities:
            raise ValueError('Each step requires a teacher prediction')
        labels = [(-1 if p[0] > threshold else 1) for p in probabilities]
        steps = []
        for text, label in zip(sample['steps'], labels):
            steps.append({'completions': [{'text': text, 'rating': label, 'flagged': False}],
                          'chosen_completion': 0 if label == 1 else None, 'human_completion': None})
            if label == -1:
                break
        yield {'question': {'problem': sample['problem'], 'pre_generated_steps': sample['steps'],
                            'pre_generated_answer': sample.get('answer'),
                            'ground_truth_answer': sample.get('ground_truth_answer')},
               'label': {'steps': steps, 'finish_reason': 'found_error' if -1 in labels else 'solution'},
               'teacher_outcome': -1 not in labels,
               'synthetic': {'threshold': threshold, 'teacher': sample.get('teacher', 'local PRM checkpoint')}}


def active_select(samples, budget, wrong_fraction=0.8, per_problem=True):
    """80% highest-scoring wrong answers + 20% highest-scoring remaining samples.

    When too few wrong answers exist, fill remaining capacity by score. Never
    duplicate samples. Intended for TRAINING pools, never test-set selection.
    """
    if budget < 1 or not 0 <= wrong_fraction <= 1:
        raise ValueError('positive budget and fraction in [0,1] required')
    groups = defaultdict(list)
    for sample in samples:
        if not math.isfinite(sample['prm_score']) or not isinstance(sample['is_correct'], bool):
            raise ValueError('finite PRM score and boolean is_correct required')
        groups[sample['problem'] if per_problem else 'all'].append(sample)
    selected = []
    for group in groups.values():
        ranked = sorted(enumerate(group), key=lambda pair: (-pair[1]['prm_score'], pair[0]))
        wrong = [(i, s) for i, s in ranked if not s['is_correct']][:math.floor(budget*wrong_fraction)]
        used = {i for i, _ in wrong}
        rest = [(i, s) for i, s in ranked if i not in used][:max(0, budget-len(wrong))]
        selected.extend(s for _, s in wrong + rest)
    return selected
