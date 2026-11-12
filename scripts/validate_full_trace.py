"""Audit complete local evidence, including every saved AdamW transition."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from prm800k.data import step_examples
from prm800k.evaluation import evaluate, group_samples
from prm800k.full_experiment import UsableSource
from prm800k.io import read_jsonl, sha256, write_json
from prm800k.model import ModelConfig, RewardModel, encode, load_model
from prm800k.scoring import aggregate, score_candidates
from prm800k.tracing import load_tensor_archive


def check_manifest(root, manifest):
    for relative, receipt in manifest.items():
        assert sha256(root/relative) == (receipt['sha256'] if isinstance(receipt, dict) else receipt), relative
    return len(manifest)


def validate_dataset(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    count = 0
    for name, source in manifest['sources'].items():
        assert sha256(root/source['raw_file']) == source['expected_sha256'] == source['raw_sha256']
        assert sha256(root/source['index_file']) == source['index_sha256']
        raw = iter(read_jsonl(root/source['raw_file']))
        seen = 0
        for entry in read_jsonl(root/source['index_file']):
            assert entry['record_index'] == seen
            record = root/entry['path']
            assert sha256(record) == entry['sha256']
            assert json.loads(record.read_text()) == next(raw)
            seen += 1
        assert next(raw, None) is None and seen == source['records']
        count += seen
        print(f'Verified {name}: {seen} individual JSON records', flush=True)
    assert count == manifest['total_records']
    return count


def audit_training(trace_path, metadata, raw_paths=None):
    trace_path = Path(trace_path)
    status = json.loads((trace_path/'status.json').read_text())
    assert status['state'] == 'complete'
    selected = next(read_jsonl(trace_path/'selected_examples.json.gz'))
    labels = {}
    for split in ('train', 'validation'):
        # Full-local deliberately uses no reservoir cap: stream coverage is exact.
        expected = None
        if raw_paths:
            source = UsableSource(raw_paths, split, metadata['seed'])
            expected = iter(step_examples(source))
        count = 0
        for row in read_jsonl(trace_path/f'{split}_examples.jsonl.gz'):
            assert row['id'] == f'{split}:{count}'
            text = row['text_and_provenance']
            if text and 'step' in text:
                tokens, start, truncated = encode(text['problem'], text['prefix']+[text['step']], metadata['config']['context'])
                assert tokens == row['token_ids'] and start == row['solution_start']
                assert bool(truncated) == row['truncated_solution'] and row['class_target'] == text['rating']+1
            if expected is not None:
                original = next(expected)
                assert all(text[key] == getattr(original, key) for key in ('problem','prefix','step','rating','source'))
            labels[row['id']] = row['class_target']
            count += 1
        if expected is not None:
            assert next(expected, None) is None
            assert len(selected[split]) == count
    model = RewardModel(ModelConfig(**metadata['config']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=metadata['learning_rate'], weight_decay=.01)
    updates = metadata['optimizer_steps']
    assert len(list((trace_path/'states').glob('*.pt.gz'))) == updates+1
    previous = load_tensor_archive(trace_path/'states/update_000000.pt.gz')
    coverage = Counter()
    max_delta = 0.0
    for update in range(1, updates+1):
        current = load_tensor_archive(trace_path/'states'/f'update_{update:06d}.pt.gz')
        batch = next(read_jsonl(trace_path/'batches'/f'update_{update:06d}.json.gz'))
        assert current['batch'] == batch and current['update'] == update
        assert current['epoch'] == batch['epoch']
        for identity, target in zip(batch['example_ids'], batch['supervised_targets']):
            assert all(t == labels[identity] for t in target)
            coverage[identity] += 1
        logits = np.array(batch['endpoint_logits'], dtype=np.float64)
        logits -= logits.max(-1, keepdims=True)
        probabilities = np.exp(logits)/np.exp(logits).sum(-1, keepdims=True)
        np.testing.assert_allclose(probabilities, batch['endpoint_probabilities'], atol=1e-7)
        losses = [loss for item in batch['token_cross_entropies_recomputed_on_cpu'] for loss in item]
        assert abs(np.mean(losses)-batch['batch_loss']) < 1e-5
        supervised = np.concatenate(batch['supervised_logits']).astype(np.float64)
        targets = np.concatenate(batch['supervised_targets'])
        supervised -= supervised.max(-1, keepdims=True)
        nll = np.log(np.exp(supervised).sum(-1))-supervised[np.arange(len(targets)), targets]
        np.testing.assert_allclose(nll, losses, atol=1e-6)
        norm = torch.stack([g.norm() for g in current['gradients_before_clip'].values()]).norm().item()
        assert abs(norm-batch['gradient_norm_before_clip']) < 1e-5*max(1, norm)
        factor = min(1.0, 1.0/(batch['gradient_norm_before_clip']+1e-6))
        for name, gradient in current['gradients_before_clip'].items():
            torch.testing.assert_close(gradient*factor, current['gradients_after_clip'][name], atol=1e-6, rtol=1e-5)
        # Replay AdamW from the preceding persisted state and actual clipped gradients.
        model.load_state_dict(previous['state_dict'])
        optimizer.load_state_dict(previous['optimizer_state_dict'])
        for name, parameter in model.named_parameters():
            parameter.grad = current['gradients_after_clip'][name]
        optimizer.step()
        for name, parameter in model.state_dict().items():
            delta = (parameter-current['state_dict'][name]).abs().max().item()
            max_delta = max(max_delta, delta)
            torch.testing.assert_close(parameter, current['state_dict'][name], atol=5e-6, rtol=1e-5)
        previous = current
        if update % 1000 == 0:
            print(f'Replayed {update}/{updates} optimizer transitions', flush=True)
    assert set(coverage) == set(selected['train'])
    assert set(coverage.values()) == {metadata['epochs']}
    for epoch in range(1, metadata['epochs']+1):
        seen = Counter()
        confusion = np.zeros((metadata['config']['classes'], metadata['config']['classes']), dtype=int)
        for batch in read_jsonl(trace_path/'validation'/f'epoch_{epoch:03d}.jsonl.gz'):
            predictions = np.array(batch['endpoint_logits']).argmax(-1)
            for identity, prediction in zip(batch['example_ids'], predictions):
                seen[identity] += 1
                confusion[labels[identity], prediction] += 1
        assert set(seen) == set(selected['validation']) and set(seen.values()) == {1}
        assert confusion.tolist() == metadata['history'][epoch-1]['confusion_matrix']
    return {'optimizer_transitions_replayed': updates, 'maximum_cpu_replay_parameter_difference': max_delta,
            'training_examples': len(selected['train']), 'validation_examples': len(selected['validation']),
            'epoch_coverage_complete': True, 'validation_confusion_recomputed': True}


def audit_inference(root):
    root = Path(root)
    samples = list(read_jsonl(root/'scored.jsonl.gz'))
    metadata = json.loads((root/'prm_training.json').read_text())
    inference = root/'details/inference'
    assert len(list((inference/'candidates').glob('*.json.gz'))) == len(samples)
    steps = 0
    for index, sample in enumerate(samples):
        record = next(read_jsonl(inference/'candidates'/f'{index:07d}.json.gz'))
        assert record['candidate']['id'] == sample['id']
        for i, entry in enumerate(record['prm_steps']):
            assert entry['steps'] == sample['steps'][:i+1]
            tokens, start, truncated = encode(sample['problem'], entry['steps'], metadata['config']['context'])
            assert tokens == entry['token_ids'] and start == entry['solution_start'] and truncated == entry['truncated_solution']
            np.testing.assert_allclose(torch.tensor(entry['logits']).softmax(-1), entry['probabilities'], atol=1e-7)
            assert entry['probabilities'] == sample['step_probabilities'][i]
            assert entry['cumulative_prm_score'] == aggregate(sample['step_probabilities'][:i+1])
            steps += 1
        assert record['reductions']['prm_score'] == sample['prm_score']
    model, _ = load_model(root/'prm.pt', 'cpu')
    reloaded = score_candidates(samples, model, batch_size=64)
    maximum = max(float(np.max(np.abs(np.array(a['step_probabilities'])-np.array(b['step_probabilities']))))
                  for a, b in zip(samples, reloaded))
    assert maximum < 1e-4
    for directory, min_candidates, ns in [('evaluation_all', 1, [1]), ('evaluation_at_least_four', 4, [1,2,4])]:
        groups = [(p, s) for p, s in sorted(group_samples(samples).items()) if len(s) >= min_candidates]
        if not groups:
            continue
        saved = json.loads((root/'details'/directory/'evaluation.json').read_text())
        recomputed = evaluate(groups, ns, methods=['prm'], seed=saved['seed'])
        assert saved['methods'] == recomputed['methods']
        for file in (root/'details'/directory/'problems').glob('*.json.gz'):
            record = next(read_jsonl(file))
            candidates = record['candidates']
            draws = record['monte_carlo']['draw_indices']
            results = record['monte_carlo']['methods']['prm']
            for trial, order in enumerate(draws):
                for j, n in enumerate(ns):
                    eligible = [i for i in order[:n] if i<len(candidates) and candidates[i]['answer'] is not None]
                    winner = max(eligible, key=lambda i: candidates[i]['scores']['prm']) if eligible else -1
                    assert winner == results['winner_indices'][trial][j]
                    assert bool(results['outcomes'][trial][j]) == (winner>=0 and candidates[winner]['is_correct'])
    return {'test_candidates_recomputed_on_cpu': len(samples), 'test_steps_checked': steps,
            'maximum_cpu_mps_probability_difference': maximum, 'all_selection_draws_replayed': True}


def main():
    # Tiny per-parameter checks are faster without CPU thread-pool overhead.
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='experiments/full-data-local')
    parser.add_argument('--dataset', default='datasets/prm800k')
    args = parser.parse_args()
    root = Path(args.run)
    start = time.perf_counter()
    counts = validate_dataset(args.dataset)
    artifacts = check_manifest(root, json.loads((root/'artifacts.json').read_text()))
    for relative, digest in json.loads((root/'source_hashes.json').read_text()).items():
        assert sha256(root/'source_snapshot'/relative) == digest
        current_source = Path(__file__).resolve().parents[1]/'prm800k'/relative
        assert sha256(current_source) == digest, f'Use the recorded source revision when validating: {relative}'
    metadata = json.loads((root/'prm_training.json').read_text())
    paths = [Path(args.dataset)/'raw'/f'{phase}_train.jsonl' for phase in ('phase1','phase2')]
    result = {'individual_raw_json_records_verified': counts, 'artifact_hashes_verified': artifacts,
              'training': audit_training(root/'details/training', metadata, paths),
              'inference_and_evaluation': audit_inference(root), 'seconds': time.perf_counter()-start}
    write_json(root/'validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
