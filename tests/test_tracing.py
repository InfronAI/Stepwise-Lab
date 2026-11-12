import gzip
import json

import numpy as np
import torch

from prm800k.data import StepExample
from prm800k.evaluation import evaluate
from prm800k.io import read_jsonl
from prm800k.model import ModelConfig, RewardModel
from prm800k.scoring import score_candidates
from prm800k.tracing import load_tensor_archive
from prm800k.training import train_steps


def test_trace_preserves_training_and_replays_each_optimizer_update(tmp_path):
    examples = [StepExample('p', [], 'a', 1, 'a'), StepExample('p', ['a'], 'b', -1, 'b')]
    validation = [StepExample('v', [], 'c', 0, 'c')]
    config = ModelConfig(width=16, layers=1, context=32)
    plain, _ = train_steps(examples, validation, tmp_path/'plain', device='cpu', epochs=2, batch_size=1, config=config)
    traced, meta = train_steps(examples, validation, tmp_path/'traced', device='cpu', epochs=2, batch_size=1,
                              config=config, trace_dir=tmp_path/'trace')
    assert meta['trace']['checkpoint_count'] == 5
    for name, tensor in plain.state_dict().items():
        assert torch.equal(tensor, traced.state_dict()[name])
    for update in range(1, 5):
        before = load_tensor_archive(tmp_path/'trace/states'/f'update_{update-1:06d}.pt.gz')
        after = load_tensor_archive(tmp_path/'trace/states'/f'update_{update:06d}.pt.gz')
        model = RewardModel(config)
        model.load_state_dict(before['state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        optimizer.load_state_dict(before['optimizer_state_dict'])
        for name, parameter in model.named_parameters():
            parameter.grad = after['gradients_after_clip'][name]
        optimizer.step()
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, after['state_dict'][name])
    records = list(read_jsonl(tmp_path/'trace/train_examples.jsonl.gz'))
    assert [r['text_and_provenance']['source'] for r in records] == ['a', 'b']
    assert len(list(read_jsonl(tmp_path/'trace/validation/epoch_002.jsonl.gz'))) == 1


def test_inference_and_selection_trace_matches_actual_outputs(tmp_path):
    model = RewardModel(ModelConfig(width=16, layers=1, context=32))
    candidates = [{'id':'a', 'problem':'p', 'steps':['one','two'], 'answer':'1', 'is_correct':True},
                  {'id':'b', 'problem':'p', 'steps':['bad'], 'answer':None, 'is_correct':False}]
    expected = score_candidates(candidates, model)
    actual = score_candidates(candidates, model, trace_dir=tmp_path/'inference')
    assert actual == expected
    record = list(read_jsonl(tmp_path/'inference/candidates/0000000.json.gz'))[0]
    assert record['prm_steps'][0]['steps'] == ['one']
    assert record['prm_steps'][-1]['cumulative_prm_score'] == actual[0]['prm_score']
    for step in record['prm_steps']:
        np.testing.assert_allclose(torch.tensor(step['logits']).softmax(-1), step['probabilities'], atol=1e-7)
    methods = ['prm','majority','random','oracle']
    plain = evaluate([('p', actual)], [1,2,3], population=3, methods=methods)
    traced = evaluate([('p', actual)], [1,2,3], population=3, methods=methods, trace_dir=tmp_path/'evaluation')
    assert plain['methods'] == traced['methods']
    trace = list(read_jsonl(tmp_path/'evaluation/problems/000001.json.gz'))[0]
    for method, records in trace['monte_carlo']['methods'].items():
        outcomes = np.array(records['outcomes'])
        assert np.all(outcomes[np.array(records['winner_indices']) == -1] == 0)
        np.testing.assert_allclose(outcomes.mean(0), traced['methods'][method]['mean'])


def test_materialization_preserves_each_original_record(tmp_path, monkeypatch):
    from prm800k import datasets
    raw = tmp_path/'source.jsonl'
    rows = [{'question':{'problem':'题目'}, 'label':{'finish_reason':'solution','steps':[]}},
            {'question':{'problem':'other'}, 'label':{'finish_reason':'give_up','steps':[]}}]
    raw.write_text('\n'.join(json.dumps(row) for row in rows))
    monkeypatch.setattr(datasets, 'LABEL_ASSETS', ('fixture',))
    monkeypatch.setattr(datasets, 'download', lambda name, directory: raw)
    monkeypatch.setattr(datasets, 'ASSETS', {'fixture': ('unused', 'not-a-real-asset', 0)})
    # Place source underneath output to match the production download contract.
    manifest = datasets.materialize(tmp_path)
    assert manifest['total_records'] == 2
    assert manifest['sources']['fixture']['usable_records'] == 1
    index = list(read_jsonl(tmp_path/'indexes/fixture.jsonl.gz'))
    assert [json.loads((tmp_path/r['path']).read_text()) for r in index] == rows
    with gzip.open(tmp_path/'indexes/fixture.jsonl.gz', 'rt') as stream:
        assert len(stream.readlines()) == 2
