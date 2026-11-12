from copy import deepcopy

import pytest

from prm800k.annotation import AnnotationSession, phase1_examples, phase2_examples, replay_annotation, teacher_examples
from prm800k.data import step_examples


def phase1():
    return {'question': {'problem': 'p1'}, 'label': {'finish_reason': 'solution', 'steps': [
        {'completions': [{'text': 'good', 'rating': 1}, {'text': 'bad', 'rating': -1}], 'chosen_completion': 0},
        {'completions': [{'text': 'bad again', 'rating': -1}], 'chosen_completion': None,
         'human_completion': {'text': 'repair', 'rating': None, 'source': 'human'}},
        {'completions': [{'text': 'done', 'rating': 1}], 'chosen_completion': 0}]}}


def phase2():
    return {'generation': 1, 'question': {'problem': 'p2', 'pre_generated_steps': ['correct', 'mistake', 'unlabeled']},
            'label': {'finish_reason': 'found_error', 'steps': [
                {'completions': [{'text': 'correct', 'rating': 1}], 'chosen_completion': 0},
                {'completions': [{'text': 'mistake', 'rating': -1}, {'text': 'alternative', 'rating': 1}], 'chosen_completion': None}]}}


def test_phase1_policy_preserves_branch_context_and_human_repair():
    all_examples = list(phase1_examples([phase1()]))
    chosen = list(phase1_examples([phase1()], 'chosen'))
    assert [(e.step, e.prefix) for e in chosen] == [('good', []), ('repair', ['good']), ('done', ['good', 'repair'])]
    assert len(all_examples) == 5
    assert len({e.source for e in all_examples}) == 5
    assert [e.rating for e in chosen] == [1, 1, 1]
    legacy = list(step_examples([phase1()]))
    assert [(e.prefix, e.step, e.rating) for e in all_examples] == [(e.prefix, e.step, e.rating) for e in legacy]


def test_phase2_mask_is_only_terminal_error_and_never_the_tail():
    included = list(phase2_examples([phase2()]))
    masked = list(phase2_examples([phase2()], False))
    assert [e.step for e in included] == ['correct', 'mistake']
    assert [e.rating for e in included] == [1, -1]
    assert [e.step for e in masked] == ['correct']
    assert included[0] == masked[0]
    assert included[-1].prefix == ['correct']


def test_phase2_stops_even_if_invalid_later_labels_are_supplied():
    row = phase2()
    row['label']['steps'].append({'completions': [{'text': 'unlabeled', 'rating': 1}]})
    assert len(list(phase2_examples([row]))) == 2


def test_phase2_alignment_and_primary_identity_despite_duplicate_text():
    row = phase2()
    row['label']['steps'][0]['completions'][0]['text'] = 'different'
    with pytest.raises(ValueError, match='does not match'):
        list(phase2_examples([row]))
    row = phase2()
    row['label']['steps'][0]['completions'].append({'text': 'correct', 'rating': -1})
    assert [e.rating for e in phase2_examples([row])] == [1, -1]
    row['label']['steps'][0]['completions'][0]['rating'] = None
    assert list(phase2_examples([row])) == []  # A sibling cannot fill a missing primary label.


def test_synthetic_full_and_first_error_keep_same_labels_on_overlap():
    sample = {'id': 'teacher:1', 'problem': 'p', 'steps': ['a', 'b', 'c'],
              'step_probabilities': [[.1, .2, .7], [.3, .2, .5], [.1, .2, .7]]}
    truncated = list(teacher_examples([sample]))
    full = list(teacher_examples([sample], stop_at_first_error=False))
    assert truncated == full[:2]
    assert full[2].prefix == ['a', 'b']
    assert full[2].label_origin == 'teacher' and full[2].after_first_error
    assert [x.rating for x in full] == [1, -1, 1]
    assert not full[1].after_first_error


def test_phase1_state_machine_and_real_schema_replay():
    trace = replay_annotation(phase1())
    assert trace['events'][1]['human_replacement']
    assert trace['events'][2]['prefix_before'] == ['good', 'repair']
    assert trace['finish_reason'] == 'solution'
    session = AnnotationSession({'problem': 'x'}, 1)
    with pytest.raises(ValueError, match='incorrect'):
        session.submit([{'text': 'wrong', 'rating': -1}], 0)
    assert session.prefix == [] and session.steps == []


def test_phase2_state_machine_cannot_resume_after_error():
    row = phase2()
    session = AnnotationSession(row['question'], 2)
    assert session.context()['next_step'] == 'correct'
    session.submit(row['label']['steps'][0]['completions'])
    session.submit(row['label']['steps'][1]['completions'])
    assert session.context()['status'] == 'found_error'
    assert len(session.export()['label']['steps']) == 2
    with pytest.raises(ValueError, match='already finished'):
        session.submit([{'text': 'unlabeled', 'rating': 1}])
    assert replay_annotation(row)['unlabeled_tail_steps'] == 1


def test_unselected_human_candidate_does_not_replace_selected_history():
    row = phase1()
    row['label']['steps'][0]['human_completion'] = {'text': 'unused human candidate'}
    trace = replay_annotation(row)
    assert trace['events'][0]['prefix_after'] == ['good']
    assert not trace['events'][0]['human_replacement']
    assert 'unused human candidate' not in [e.step for e in phase1_examples([row])]


def test_policy_inputs_are_not_mutated_and_cross_phase_is_rejected():
    a, b = phase1(), phase2()
    before = deepcopy([a, b])
    list(phase1_examples([a]))
    list(phase2_examples([b]))
    assert [a, b] == before
    with pytest.raises(ValueError, match='phase2'):
        list(phase1_examples([b]))
    with pytest.raises(ValueError, match='pregenerated'):
        list(phase2_examples([a]))


def test_equal_budget_preserves_identical_problem_quotas_and_targets():
    from collections import Counter
    from prm800k.protocol_experiments import paired_subsets
    a = phase1()
    b = deepcopy(a)
    b['question']['problem'] = 'another problem'
    left = list(phase1_examples([a, b]))
    right = list(phase1_examples([a, b], 'chosen'))
    x, y, receipt = paired_subsets(left, right, budget=5, seed=42)
    assert len(x) == len(y) == 5
    assert Counter(e.problem for e in x) == Counter(e.problem for e in y)
    assert len({e.source for e in x}) == len(x)
    assert paired_subsets(left, right, 5, 42) == (x, y, receipt)
    with pytest.raises(ValueError, match='cannot cover'):
        paired_subsets(left, right, 1, 42)


def test_common_problem_exclusions_are_explicit():
    from prm800k.protocol_experiments import paired_subsets
    row = phase2()
    empty_prefix = deepcopy(row)
    empty_prefix['question']['problem'] = 'error at start'
    empty_prefix['question']['pre_generated_steps'] = ['mistake']
    empty_prefix['label']['steps'] = [row['label']['steps'][1]]
    left = list(phase2_examples([row, empty_prefix]))
    right = list(phase2_examples([row, empty_prefix], False))
    x, y, receipt = paired_subsets(left, right, 10, 42)
    assert receipt['excluded_problems'] == [1, 0]
    assert {e.problem for e in x} == {e.problem for e in y} == {'p2'}


def test_trainer_refactor_preserves_existing_prm_updates(tmp_path):
    import torch
    from prm800k.model import ModelConfig
    from prm800k.training import train, train_steps
    tr = phase1()
    va = deepcopy(tr)
    va['question']['problem'] = 'held out'
    config = ModelConfig(width=16, layers=1, context=64)
    original, _ = train([tr], [va], tmp_path/'legacy', config=config, device='cpu', epochs=1, seed=9)
    explicit, meta = train_steps(list(phase1_examples([tr])), list(phase1_examples([va])),
                                tmp_path/'explicit', config=config, device='cpu', epochs=1, seed=9)
    for name, values in original.state_dict().items():
        assert torch.equal(values, explicit.state_dict()[name])
    assert meta['optimizer_steps'] == 1 and meta['supervised_tokens_per_epoch'] == 5


def test_teacher_split_requires_a_disjoint_provenance_receipt(tmp_path):
    from prm800k.protocol_experiments import _validate_teacher_split
    from prm800k.io import write_json, problem_id
    teacher = {'kind':'prm', 'config':{'classes':3}}
    tr, va = phase1(), phase2()
    path = tmp_path/'inputs.json'
    with pytest.raises(ValueError, match='adjacent inputs'):
        _validate_teacher_split(teacher, path, [tr], [va], [])
    write_json(path, {'train_problem_ids':[problem_id('p2')]})
    with pytest.raises(ValueError, match='incompatible'):
        _validate_teacher_split(teacher, path, [tr], [va], [])
    write_json(path, {'train_problem_ids':[problem_id('p1')]})
    assert _validate_teacher_split(teacher, path, [tr], [va], [])['teacher_overlap_with_validation_or_test'] == 0


def test_all_released_fixture_trajectories_replay_without_inventing_tail_labels():
    from pathlib import Path
    from prm800k.io import read_jsonl
    root = Path(__file__).resolve().parents[1]/'prm800k/fixtures'
    rows = list(read_jsonl(root/'train.jsonl.gz')) + list(read_jsonl(root/'test.jsonl.gz'))
    assert len(rows) == 952
    for row in rows:
        trace = replay_annotation(row)
        assert len(trace['events']) == len(row['label']['steps'])
        assert trace['finish_reason'] == row['label']['finish_reason']
        if trace['phase'] == 2:
            examples = list(phase2_examples([row]))
            assert len(examples) <= len(row['label']['steps'])
            assert all(e.rating != -1 for e in examples[:-1])
