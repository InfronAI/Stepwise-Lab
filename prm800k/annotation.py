"""Explicit annotation flows and traceable policy projections.

Human phase-2 labels never extend past the first error. Full-trajectory labels
are only produced by the separately identified synthetic teacher projection.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json

from .data import StepExample, human_text, usable


def trajectory_id(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def annotation_phase(row):
    return 2 if row['question'].get('pre_generated_steps') is not None else 1


def valid_label(completion):
    return type(completion.get('rating')) is int and completion['rating'] in (-1, 0, 1) and not completion.get('flagged')


@dataclass
class AnnotationExample(StepExample):
    trajectory_id: str
    phase: str
    step_index: int
    completion_index: int | None
    label_origin: str
    after_first_error: bool = False


def make_example(row, prefix, text, rating, index, completion, origin='human', after_error=False):
    identity = trajectory_id(row)
    return AnnotationExample(row['question']['problem'], list(prefix), text, rating,
                             f'{identity}:{index}:{completion if completion is not None else "human"}',
                             identity, f'phase{annotation_phase(row)}', index, completion, origin, after_error)


def phase1_examples(rows, policy='all'):
    """Project interactive branches onto all alternatives or the selected path.

    Human replacements are labels in both policies and extend the SAME chosen
    history. Unchosen siblings never enter later prefixes.
    """
    if policy not in ('all', 'chosen'):
        raise ValueError('phase1 policy must be all or chosen')
    for row in rows:
        if annotation_phase(row) != 1:
            raise ValueError('phase1 projection received a phase2 trajectory')
        if not usable(row):
            continue
        prefix = []
        for index, step in enumerate(row['label']['steps']):
            chosen = step.get('chosen_completion')
            completions = step['completions']
            if chosen is not None and (type(chosen) is not int or not 0 <= chosen < len(completions)):
                raise ValueError('chosen_completion index out of bounds')
            for ci, completion in enumerate(completions):
                if valid_label(completion) and (policy == 'all' or ci == chosen):
                    yield make_example(row, prefix, completion['text'], completion['rating'], index, ci)
            human = human_text(step.get('human_completion'))
            if chosen is None:
                if not human:
                    break
                yield make_example(row, prefix, human, 1, index, None, 'human_replacement')
                prefix.append(human)
            else:
                completion = completions[chosen]
                if not valid_label(completion) or completion['rating'] == -1:
                    break
                prefix.append(completion['text'])


def phase2_examples(rows, include_first_error=True):
    """Only the original pregenerated path, aligned by exact step text.

    Additional alternatives at the error position are not part of the original
    path and are excluded in BOTH arms of this experiment. Unlabeled tails are
    never assigned labels. The include_first_error=False arm masks that single
    terminal supervision target, not its earlier context.
    """
    for row in rows:
        if annotation_phase(row) != 2:
            raise ValueError('phase2 projection requires pregenerated steps')
        if not usable(row):
            continue
        original = row['question']['pre_generated_steps']
        for index, step in enumerate(row['label']['steps']):
            if index >= len(original):
                raise ValueError('More labeled positions than original steps')
            # The release stores the original as completion 0. Siblings may
            # repeat its text with another rating; their identity is distinct.
            if not step['completions'] or step['completions'][0]['text'] != original[index]:
                raise ValueError(f'Original step does not match annotation position {index}')
            completion = step['completions'][0]
            if not valid_label(completion):
                break
            rating = completion['rating']
            if rating != -1 or include_first_error:
                yield make_example(row, original[:index], original[index], rating, index, 0)
            if rating == -1:
                break


def teacher_examples(samples, *, stop_at_first_error=True, threshold=0.2):
    """Paired synthetic policies from the same frozen teacher predictions.

    Full-trajectory targets after a teacher error are SYNTHETIC, not newly
    observed human labels. Prefixes continue along the original generated path.
    """
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0,1]')
    for sample in samples:
        steps, probabilities = sample['steps'], sample['step_probabilities']
        if len(steps) != len(probabilities) or not steps:
            raise ValueError('A teacher distribution is required for every step')
        identity = str(sample['id'])
        after_error = False
        for index, (step, probs) in enumerate(zip(steps, probabilities)):
            if len(probs) != 3 or any(not 0 <= x <= 1 for x in probs) or abs(sum(probs)-1) > 1e-5:
                raise ValueError('Expected normalized teacher class probabilities')
            rating = -1 if probs[0] > threshold else 1
            yield AnnotationExample(sample['problem'], steps[:index], step, rating,
                                    f'{identity}:{index}:teacher', identity, 'synthetic_phase2', index, 0,
                                    'teacher', after_error)
            if rating == -1:
                if stop_at_first_error:
                    break
                after_error = True


class AnnotationSession:
    """Small state machine for interactive labeling or deterministic replay.

    Phase 1: submit reviewed alternatives, then advance only on the chosen or
    human replacement step. Phase 2: review the fixed next step, stop on -1.
    Calling submit after completion is an error. No labels are inferred for
    an unreviewed step. This class does not call a generator or impersonate a
    human labeler; callers supply candidates and judgments explicitly.
    """
    def __init__(self, question, phase):
        if phase not in (1, 2):
            raise ValueError('phase must be 1 or 2')
        self.phase = phase
        self.question = deepcopy(question)
        if phase == 2 and not question.get('pre_generated_steps'):
            raise ValueError('phase2 requires a nonempty fixed trajectory')
        self.steps = []
        self.prefix = []
        self.finish_reason = None
        self.events = []

    def context(self):
        return {'problem': self.question['problem'], 'prefix': list(self.prefix),
                'next_step': self.question['pre_generated_steps'][len(self.steps)]
                if self.phase == 2 and self.finish_reason is None else None,
                'status': self.finish_reason or 'awaiting_review'}

    def submit(self, completions, chosen_completion=None, human_completion=None, *, finish_reason=None):
        if self.finish_reason is not None:
            raise ValueError('Annotation session is already finished')
        if not completions or any(not isinstance(c.get('text'), str) for c in completions):
            raise ValueError('At least one textual candidate is required')
        if any(c.get('rating') is not None and (type(c['rating']) is not int or c['rating'] not in (-1, 0, 1)) for c in completions):
            raise ValueError('Ratings must be -1, 0, 1 or null')
        if finish_reason not in (None, 'solution', 'give_up', 'bad_problem'):
            raise ValueError('Unsupported explicit finish reason')
        before = list(self.prefix)
        human = human_text(human_completion)
        if self.phase == 1:
            if chosen_completion is not None:
                if type(chosen_completion) is not int or not 0 <= chosen_completion < len(completions):
                    raise ValueError('chosen_completion index out of bounds')
                selected = completions[chosen_completion]
                if not valid_label(selected) or selected['rating'] == -1:
                    raise ValueError('Cannot continue phase1 along an incorrect or unreviewed candidate')
                self.prefix.append(selected['text'])
            elif human:
                self.prefix.append(human)
            elif finish_reason is None:
                raise ValueError('Choose a candidate, supply a replacement, or finish the task')
            self.finish_reason = finish_reason
        else:
            index = len(self.steps)
            if human_completion is not None or completions[0]['text'] != self.question['pre_generated_steps'][index]:
                raise ValueError('phase2 must review the next original step without a human replacement')
            primary = completions[0]
            if not valid_label(primary):
                raise ValueError('An unambiguous primary-step rating is required')
            if primary['rating'] == -1:
                chosen_completion = None
                self.finish_reason = 'found_error'
            else:
                chosen_completion = 0
                self.prefix.append(primary['text'])
                self.finish_reason = finish_reason
                if index + 1 == len(self.question['pre_generated_steps']):
                    self.finish_reason = self.finish_reason or 'solution'
        self.steps.append({'completions': deepcopy(completions), 'chosen_completion': chosen_completion,
                           'human_completion': deepcopy(human_completion)})
        self.events.append({'step_index': len(self.steps)-1, 'prefix_before': before,
                            'candidate_count': len(completions), 'chosen_completion': chosen_completion,
                            'human_replacement': bool(human) and chosen_completion is None, 'prefix_after': list(self.prefix),
                            'state': self.finish_reason or 'awaiting_review'})
        return self.context()

    def export(self):
        if self.finish_reason is None:
            raise ValueError('Finish annotation before exporting a training trajectory')
        return {'question': deepcopy(self.question), 'generation': None if self.phase == 1 else 0,
                'is_quality_control_question': False, 'is_initial_screening_question': False,
                'label': {'steps': deepcopy(self.steps), 'finish_reason': self.finish_reason}}


def replay_annotation(row):
    """Replay a released usable record, returning the explicit transition trace."""
    phase = annotation_phase(row)
    session = AnnotationSession(row['question'], phase)
    for index, step in enumerate(row['label']['steps']):
        terminal = row['label']['finish_reason'] if index == len(row['label']['steps'])-1 else None
        # A negative phase2 primary sets found_error itself.
        explicit = terminal if terminal in ('solution', 'give_up', 'bad_problem') else None
        session.submit(step['completions'], step.get('chosen_completion'), step.get('human_completion'), finish_reason=explicit)
    result = session.export()
    if result['label']['finish_reason'] != row['label']['finish_reason']:
        raise ValueError('Replay does not agree with the released finish reason')
    return {'trajectory_id': trajectory_id(row), 'phase': phase, 'question': row['question']['problem'],
            'events': session.events, 'finish_reason': session.finish_reason,
            'unlabeled_tail_steps': max(0, len(row['question'].get('pre_generated_steps') or [])-len(session.steps))}


def example_dict(example):
    return asdict(example)
