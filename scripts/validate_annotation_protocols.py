"""Independently check saved protocol inputs, pairing, predictions and metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from prm800k.annotation import example_dict, phase1_examples, phase2_examples, teacher_examples, annotation_phase
from prm800k.evaluation import evaluate, group_samples
from prm800k.io import read_jsonl, sha256
from prm800k.model import load_model
from prm800k.protocol_experiments import CONDITIONS
from prm800k.scoring import score_candidates


def validate(directory):
    directory = Path(directory)
    def read(name):
        return json.loads((directory/name).read_text())
    manifest = read('artifacts.json')
    for filename, expected in manifest.items():
        assert sha256(directory/filename) == expected, filename
    source = Path(__file__).resolve().parents[1]/'prm800k'
    for filename, expected in read('source_hashes.json').items():
        assert sha256(source/filename) == expected, f'Source changed: {filename}'
    summary, splits = read('summary.json'), read('inputs.json')
    train_ids, val_ids, test_ids = [set(splits[f'{s}_problem_ids']) for s in ('train', 'validation', 'test')]
    assert not train_ids & val_ids and not train_ids & test_ids and not val_ids & test_ids
    raw = list(read_jsonl(directory/'train_trajectories.jsonl.gz'))
    sources = {}
    if 'phase1_branches' in summary['config']['families']:
        rows = [r for r in raw if annotation_phase(r) == 1]
        sources['phase1_branches'] = (list(phase1_examples(rows)), list(phase1_examples(rows, 'chosen')))
    if 'phase2_error_label' in summary['config']['families']:
        rows = [r for r in raw if annotation_phase(r) == 2]
        sources['phase2_error_label'] = (list(phase2_examples(rows)), list(phase2_examples(rows, False)))
    if 'synthetic_horizon' in summary['config']['families']:
        rows = list(read_jsonl(directory/'teacher_train_predictions.jsonl.gz'))
        threshold = summary['config']['teacher_threshold']
        sources['synthetic_horizon'] = (list(teacher_examples(rows, threshold=threshold)),
            list(teacher_examples(rows, stop_at_first_error=False, threshold=threshold)))
    original_candidates = list(read_jsonl(directory/'test_candidates.jsonl.gz'))
    test_problems = {s['problem'] for s in original_candidates}
    checked, maximum_delta = 0, 0.0
    for family, pools in sources.items():
        lookup = {condition: {e.source: example_dict(e) for e in pool} for condition, pool in zip(CONDITIONS[family], pools)}
        validation = list(read_jsonl(directory/family/'validation_examples.jsonl.gz'))
        validation_problems = {e['problem'] for e in validation}
        for seed in summary['config']['seeds']:
            paired = []
            quotas = []
            for condition in CONDITIONS[family]:
                run = directory/family/f'{condition}-seed{seed}'
                examples = list(read_jsonl(run/'training_examples.jsonl.gz'))
                assert len({e['source'] for e in examples}) == len(examples)
                for example in examples:
                    assert example == lookup[condition][example['source']]
                    assert example['problem'] not in test_problems | validation_problems
                    assert example['label_origin'] == 'teacher' or not example['after_first_error']
                quotas.append(Counter(e['problem'] for e in examples))
                metadata = json.loads((run/'prm_training.json').read_text())
                assert metadata['training_examples'] == len(examples)
                assert metadata['training_examples_sha256'] == sha256(run/'training_examples.jsonl.gz')
                paired.append(metadata)
                scored = list(read_jsonl(run/'scored.jsonl.gz'))
                assert len(scored) == len(original_candidates)
                for original, prediction in zip(original_candidates, scored):
                    assert all(prediction[key] == value for key, value in original.items())
                recomputed = evaluate(sorted(group_samples(scored).items()), summary['ns'], methods=['prm'], trials=1)
                saved = json.loads((run/'evaluation.json').read_text())
                np.testing.assert_allclose(recomputed['methods']['prm']['exact_mean'], saved['methods']['prm']['exact_mean'], atol=1e-12)
                model, loaded_metadata = load_model(run/'prm.pt', 'cpu')
                assert metadata == loaded_metadata
                reloaded = score_candidates(original_candidates[:1], model)[0]
                delta = np.max(np.abs(np.array(reloaded['step_probabilities'])-np.array(scored[0]['step_probabilities'])))
                maximum_delta = max(maximum_delta, float(delta))
                assert delta < 1e-4, (run, delta)
                checked += 1
            assert quotas[0] == quotas[1]
            for key in ('seed', 'config', 'initial_backbone_sha256', 'optimizer_steps',
                        'supervised_tokens_per_epoch', 'validation_examples_sha256'):
                assert paired[0][key] == paired[1][key]
    return {'directory': str(directory), 'checked_runs': checked, 'verified_artifact_hashes': len(manifest),
            'source_hashes_match': True, 'problem_split_overlap': 0,
            'exact_metrics_recomputed': True, 'paired_problem_quotas_match': True,
            'all_training_examples_match_source_projections': True,
            'checkpoint_cpu_reload_max_probability_difference': maximum_delta}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='experiments/annotation-protocols')
    print(json.dumps(validate(parser.parse_args().directory), indent=2))
