import itertools
import math

import numpy as np
import pytest

from prm800k.evaluation import exact_problem, evaluate, build_index, indexed_groups, probability_avoid
from prm800k.io import write_jsonl


def sample(score, correct, answer='a'):
    return {'problem': 'p', 'prm_score': score, 'orm_score': 1-score, 'is_correct': correct, 'answer': answer}


def enumerate_expected(samples, n, population):
    padded = samples + [None]*(population-len(samples))
    values = []
    for indices in itertools.combinations(range(population), n):
        valid = [padded[i] for i in indices if padded[i] is not None and padded[i]['answer'] is not None]
        if not valid:
            values.append(0)
        else:
            best = max(s['prm_score'] for s in valid)
            tied = [s for s in valid if s['prm_score'] == best]
            values.append(sum(s['is_correct'] for s in tied)/len(tied))
    return sum(values)/len(values)


@pytest.mark.parametrize('population', [4, 5, 7])
def test_exact_matches_exhaustive_subsets_including_ties_and_missing(population):
    samples = [sample(.8, True), sample(.8, False), sample(.2, True), sample(.9, True, None)]
    ns = list(range(1, population+1))
    expected = [enumerate_expected(samples, n, population) for n in ns]
    assert np.allclose(exact_problem(samples, ns, population, 'prm'), expected, atol=1e-12)


def test_avoid_probability():
    for population in range(1, 12):
        for excluded in range(population+1):
            for n in range(1, population+1):
                exact = math.comb(population-excluded, n)/math.comb(population, n) if population-excluded >= n else 0
                assert probability_avoid(population, excluded, n) == pytest.approx(exact)


def test_monte_carlo_seed_and_exact_agreement():
    groups = [('p',[sample(.8, True), sample(.2, False)])]
    a = evaluate(groups, [1,2], population=4, trials=10000)
    b = evaluate(groups, [1,2], population=4, trials=10000)
    assert a['methods'] == b['methods']
    for method in ['prm', 'orm']:
        assert np.allclose(a['methods'][method]['mean'], a['methods'][method]['exact_mean'], atol=.015)


def test_empty_and_population_errors():
    with pytest.raises(ValueError, match='No problems'):
        evaluate([], [1])
    with pytest.raises(ValueError, match='population'):
        exact_problem([sample(1,True)], [1], 0, 'prm')
    with pytest.raises(ValueError, match='Each N'):
        exact_problem([sample(1,True)], [2], 1, 'prm')


def test_disk_index_parity(tmp_path):
    samples = [sample(.8, True), sample(.2, False)]
    source = tmp_path/'samples.jsonl'
    index = tmp_path/'samples.sqlite'
    write_jsonl(source, samples)
    assert build_index(source, index) == 2
    assert list(indexed_groups(index)) == [('p', samples)]


def test_missing_all_fail():
    result = evaluate([('p',[sample(1,True,None)])], [1,2], population=2)
    assert all(m['mean'] == [0,0] for m in result['methods'].values())


def test_majority_normalizes_fraction():
    samples = [sample(.1,True,'0.5'),sample(.2,True,r'\frac{1}{2}'),sample(.9,False,'9')]
    result = evaluate([('p',samples)], [3], methods=['majority'], trials=30)
    assert result['methods']['majority']['mean'] == [1]


def test_subject_and_difficulty_breakdowns(tmp_path):
    samples=[dict(sample(.8,True),subject='Algebra',level=1),dict(sample(.2,False),subject='Algebra',level=1)]
    source=tmp_path/'samples.jsonl'
    index=tmp_path/'samples.sqlite'
    write_jsonl(source,samples)
    build_index(source,index)
    result=evaluate(indexed_groups(index),[1,2],methods=['prm'])
    assert result['breakdowns']['subject']['Algebra']['problems']==1
    assert result['breakdowns']['level']['1']['methods']['prm']==[.5,1.0]
    assert result['breakdowns']['generator_difficulty']['quintile_1_hardest_first']['problems']==1
