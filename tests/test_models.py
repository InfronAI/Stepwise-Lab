import math

import pytest
import torch

from prm800k.model import RewardModel, ModelConfig, encode, load_model, save_model, seed_all
from prm800k.scoring import aggregate, predict
from prm800k.supervision import active_select, synthetic_rows
from prm800k.training import prepare
from prm800k.generation import extract_answer
from prm800k.grading.grader import grade_answer


def test_causal_no_future_leak():
    seed_all(7)
    model = RewardModel(ModelConfig(width=16,layers=1,context=64)).eval()
    tokens = torch.tensor([[257,5,6,7,8]])
    with torch.inference_mode():
        assert torch.allclose(model(tokens)[:,:3], model(tokens[:,:3]), atol=1e-6)


def test_checkpoint_prediction_roundtrip(tmp_path):
    model = RewardModel(ModelConfig(width=16,layers=1,context=64)).eval()
    path = tmp_path/'model.pt'
    save_model(path, model, {'kind':'prm'})
    loaded, meta = load_model(path, 'cpu')
    assert meta['kind'] == 'prm'
    assert predict(model, [('question',['one','two'])]) == predict(loaded, [('question',['one','two'])])


def test_context_and_unicode():
    tokens, start, truncated = encode('题目'*20, ['x'*500],64)
    assert len(tokens) == 64 and truncated
    assert start > 1
    assert all(0 <= token < 259 for token in tokens)


def test_log_product_and_neutral_policies():
    p = [[.2,.3,.5],[.1,.2,.7]]
    assert aggregate(p) == pytest.approx(math.log(.8*.9))
    assert aggregate(p,False) == pytest.approx(math.log(.5*.7))
    assert aggregate(p,reduction='minimum') == pytest.approx(math.log(.8))
    assert aggregate(p,False,'minimum') == pytest.approx(math.log(.5))
    assert math.isfinite(aggregate([[1,0,0]]*10000))
    with pytest.raises(ValueError):
        aggregate([[.5,.5,.5]])


def candidate(i, correct=False):
    return {'id':str(i),'problem':'p','steps':['a','b','c'],'answer':'2','ground_truth_answer':'2',
            'is_correct':correct,'prm_score':float(i),'step_probabilities':[[.1,.1,.8],[.3,.1,.6],[.1,.1,.8]]}


def test_synthetic_cutoff_and_teacher_outcome():
    rows = list(synthetic_rows([candidate(0)]))
    assert len(rows[0]['label']['steps']) == 2
    assert rows[0]['teacher_outcome'] is False
    assert rows[0]['label']['steps'][-1]['completions'][0]['rating'] == -1
    # ORM based on teacher and final-answer checking intentionally disagree here.
    assert next(prepare(rows,'orm-teacher',64))[2] == 0
    assert next(prepare(rows,'orm',64))[2] == 1


def test_active_80_percent_wrong_and_no_duplicates():
    pool = [candidate(i,i>=8) for i in range(10)]
    selected = active_select(pool,5,per_problem=False)
    assert len(selected) == 5
    assert sum(not s['is_correct'] for s in selected) == 4
    assert len({s['id'] for s in selected}) == 5
    assert selected[-1]['id'] == '9'


@pytest.mark.parametrize('given,truth,expected', [('0.5',r'\frac{1}{2}',True),('2','3',False),(None,'3',False),('40,000','40000',True),('(1,2)','[1,2]',False)])
def test_original_grader(given,truth,expected):
    assert grade_answer(given,truth) == expected


def test_nested_answer_extraction():
    assert extract_answer(r'Answer \boxed{\frac{1}{2}}') == r'\frac{1}{2}'
    assert extract_answer('unfinished solution') is None
    assert extract_answer('text\n# Answer\n\n42') == '42'


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple Metal device unavailable')
def test_cpu_mps_prediction_parity():
    seed_all(42)
    model=RewardModel(ModelConfig(width=16,layers=1,context=64)).eval()
    requests=[('a question',['the first step','the second step'])]
    cpu=torch.tensor(predict(model,requests))
    metal=torch.tensor(predict(model.to('mps'),requests))
    assert torch.allclose(cpu,metal,atol=1e-4,rtol=1e-4)


def test_prm_orm_start_with_identical_seeded_backbones():
    seed_all(17)
    prm=RewardModel(ModelConfig(width=16,layers=1,context=64,classes=3))
    seed_all(17)
    orm=RewardModel(ModelConfig(width=16,layers=1,context=64,classes=2))
    for key,value in prm.state_dict().items():
        if not key.startswith('head.'):
            assert torch.equal(value,orm.state_dict()[key]),key
