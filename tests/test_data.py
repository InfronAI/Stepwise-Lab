import copy
import gzip

import pytest

from prm800k.data import step_examples, split_by_problem
from prm800k.io import read_jsonl, write_jsonl, sha256


def row(problem='p'):
    return {'question': {'problem': problem}, 'label': {'finish_reason': 'solution', 'steps': [
        {'completions': [{'text': 'a', 'rating': 1}, {'text': 'b', 'rating': -1}], 'chosen_completion': 0},
        {'completions': [{'text': 'c', 'rating': 0}], 'chosen_completion': 0}]}}


def test_alternatives_share_only_chosen_history():
    examples = list(step_examples([row()]))
    assert [(x.step, x.prefix) for x in examples] == [('a', []), ('b', []), ('c', ['a'])]


def test_first_error_null_and_human_replacement():
    r = row()
    r['label']['steps'][0]['chosen_completion'] = None
    assert len(list(step_examples([r]))) == 2
    r['label']['steps'][0]['human_completion'] = 'human corrected step'
    examples = list(step_examples([r]))
    assert examples[2].rating == 1
    assert examples[-1].prefix == ['human corrected step']


def test_skip_quality_control_and_unfinished():
    r = row()
    r['is_quality_control_question'] = True
    assert not list(step_examples([r]))
    r['is_quality_control_question'] = False
    r['label']['finish_reason'] = 'give_up'
    assert not list(step_examples([r]))


def test_flagged_and_unrated():
    r = row()
    r['label']['steps'][0]['completions'][0]['flagged'] = True
    r['label']['steps'][0]['completions'][1]['rating'] = None
    assert not list(step_examples([r]))


def test_problem_split_keeps_duplicate_problems_together():
    rows = [row(str(i)) for i in range(20)] * 2
    a, b = split_by_problem(rows)
    assert not {r['question']['problem'] for r in a} & {r['question']['problem'] for r in b}
    assert len(a)+len(b) == len(rows)
    assert (a,b) == split_by_problem(rows)


def test_gzip_roundtrip_reproducible(tmp_path):
    a, b = tmp_path/'a.jsonl.gz', tmp_path/'b.jsonl.gz'
    write_jsonl(a, [row()])
    write_jsonl(b, [row()])
    assert list(read_jsonl(a)) == [row()]
    assert sha256(a) == sha256(b)
    assert gzip.open(a).read().startswith(b'{')


def test_lfs_error_and_json_line_number(tmp_path):
    p = tmp_path/'data.jsonl'
    p.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError, match='LFS pointer'):
        list(read_jsonl(p))
    p.write_text('{}\nnot json\n')
    with pytest.raises(ValueError, match=':2:'):
        list(read_jsonl(p))


def test_bad_chosen_index():
    r = copy.deepcopy(row())
    r['label']['steps'][0]['chosen_completion'] = 9
    with pytest.raises(ValueError, match='out of bounds'):
        list(step_examples([r]))


def test_actual_phase1_human_completion_object():
    r = row()
    r['label']['steps'][0]['chosen_completion'] = None
    r['label']['steps'][0]['human_completion'] = {'text':'human text','rating':None,'source':'human','flagged':False}
    examples = list(step_examples([r]))
    assert examples[2].step == 'human text'
    assert examples[2].rating == 1
    assert examples[-1].prefix == ['human text']


def test_streaming_source_is_repeatable_and_disjoint(tmp_path):
    from prm800k.data import AnnotationSource
    path=tmp_path/'rows.jsonl'
    write_jsonl(path,[row(str(i)) for i in range(100)])
    a=AnnotationSource([path],'train')
    b=AnnotationSource([path],'validation')
    assert list(a)==list(a)
    assert {r['question']['problem'] for r in a}.isdisjoint({r['question']['problem'] for r in b})
    assert len(list(a))+len(list(b))==100
