"""Train locally on every usable official training label and retain every update."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from .assets import ASSETS
from .data import AnnotationSource, candidates, usable
from .datasets import LABEL_ASSETS, materialize
from .evaluation import evaluate, group_samples
from .experiment import environment
from .io import problem_id, read_jsonl, sha256, write_json, write_jsonl
from .model import ModelConfig, load_model
from .scoring import score_candidates
from .training import train


class UsableSource(AnnotationSource):
    def __iter__(self):
        return (row for row in super().__iter__() if usable(row))


def full_experiment(output, dataset='datasets/prm800k', *, device='auto', seed=42,
                    epochs=1, context=128, width=32, batch_size=64):
    """Full data coverage with a small local model; no reservoir subsampling."""
    output, dataset = Path(output), Path(dataset)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; existing evidence is never overwritten')
    if not (dataset/'manifest.json').exists():
        materialize(dataset)
    manifest = json.loads((dataset/'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Dataset materialization is incomplete; rerun materialize-data')
    for name in LABEL_ASSETS:
        if sha256(dataset/'raw'/f'{name}.jsonl') != ASSETS[name][1]:
            raise ValueError(f'Official raw file verification failed: {name}')
    output.mkdir(parents=True)
    started = time.perf_counter()
    state = {'state': 'running', 'stages': [], 'dataset': str(dataset)}
    def stage(name):
        if state['stages']:
            state['stages'][-1]['state'] = 'complete'
            state['stages'][-1]['ended_at_unix'] = time.time()
        state['stages'].append({'name': name, 'state': 'running', 'started_at_unix': time.time()})
        write_json(output/'pipeline.json', state)
        print(f'Stage: {name}', flush=True)
    try:
        stage('verify_data_and_split_by_problem')
        paths = [dataset/'raw'/f'{phase}_train.jsonl' for phase in ('phase1', 'phase2')]
        training = UsableSource(paths, 'train', seed)
        validation = UsableSource(paths, 'validation', seed)
        train_ids = {problem_id(r['question']['problem']) for r in training}
        validation_ids = {problem_id(r['question']['problem']) for r in validation}
        testing = list(candidates(read_jsonl(dataset/'raw/phase2_test.jsonl')))
        test_ids = {problem_id(r['problem']) for r in testing}
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise ValueError('Train/validation/test problem overlap')
        write_json(output/'inputs.json', {'dataset_manifest_sha256': sha256(dataset/'manifest.json'),
            'raw_files': {n: ASSETS[n][1] for n in LABEL_ASSETS}, 'train_problem_ids': sorted(train_ids),
            'validation_problem_ids': sorted(validation_ids), 'test_problem_ids': sorted(test_ids),
            'overlap': 0, 'split': 'SHA-256 problem split: 80% training / 20% validation',
            'selection': 'every usable annotation from both official training files; no row/example cap'})
        config = ModelConfig(width=width, context=context)
        write_json(output/'config.json', {'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
                   'model': asdict(config), 'device': device, 'full_training_data': True, 'trace_every_update': True})
        source = Path(__file__).parent
        write_json(output/'source_hashes.json', {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*.py'))})
        for path in source.rglob('*.py'):
            snapshot = output/'source_snapshot'/path.relative_to(source)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
        write_jsonl(output/'test_candidates.jsonl.gz', testing)
        stage('project_encode_and_train_prm')
        _, metadata = train(training, validation, output, device=device, seed=seed, epochs=epochs,
            batch_size=batch_size, config=config, max_examples=2**63-1, trace_dir=output/'details/training')
        stage('reload_checkpoint_and_infer_every_test_step')
        # Deliberately reload the checkpoint: inference must use persisted weights.
        prm, _ = load_model(output/'prm.pt', device)
        scored = score_candidates(testing, prm, batch_size=batch_size, trace_dir=output/'details/inference')
        write_jsonl(output/'scored.jsonl.gz', scored)
        stage('rank_candidates_and_evaluate')
        groups = sorted(group_samples(scored).items())
        all_metrics = evaluate(groups, [1], methods=['prm'], seed=seed, trace_dir=output/'details/evaluation_all')
        write_json(output/'evaluation_all.json', all_metrics)
        cohort = [(p, samples) for p, samples in groups if len(samples) >= 4]
        cohort_metrics = evaluate(cohort, [1,2,4], methods=['prm'], seed=seed,
                                  trace_dir=output/'details/evaluation_at_least_four') if cohort else None
        write_json(output/'evaluation_best_of_n.json', {'eligible_problems': len(cohort), 'all_problems': len(groups),
            'rule': 'fixed cohort with >=4 labeled candidates; unlabeled generations are not invented as failures',
            'metrics': cohort_metrics})
        selections = []
        for problem, samples in groups:
            ranked = sorted(enumerate(samples), key=lambda pair: (-pair[1]['prm_score'], pair[0]))
            top = ranked[0][1]['prm_score']
            ties = [s for _, s in ranked if s['prm_score'] == top]
            selections.append({'problem': problem, 'ranked_candidate_ids': [s['id'] for _, s in ranked],
                'ranked_scores': [s['prm_score'] for _, s in ranked], 'top_tied_candidate_ids': [s['id'] for s in ties],
                'expected_correctness_uniform_top_ties': sum(s['is_correct'] for s in ties)/len(ties)})
        write_jsonl(output/'ranked_selections.jsonl.gz', selections)
        stage('write_analysis_and_checksums')
        summary = {'environment': environment(metadata['device']), 'seconds': time.perf_counter()-started,
            'training_examples': metadata['training_examples'], 'validation_examples': metadata['validation_examples'],
            'optimizer_updates': metadata['optimizer_steps'], 'test_candidates': len(scored),
            'test_step_predictions': sum(len(s['steps']) for s in scored), 'test_problems': len(groups),
            'best_of_n_cohort_problems': len(cohort),
            'available_candidate_top_selection_accuracy': sum(s['expected_correctness_uniform_top_ties'] for s in selections)/len(selections),
            'checkpoint_sha256': sha256(output/'prm.pt'), 'scope': 'Full released training-label coverage; small randomly initialized local PRM, not GPT-4 reproduction'}
        write_json(output/'summary.json', summary)
        for lang in ('en', 'zh-CN'):
            title = '完整本地训练与逐步推理' if lang == 'zh-CN' else 'Full local training and step inference'
            (output/f'ANALYSIS.{lang}.md').write_text(f'# {title}\n\n```json\n'+json.dumps(summary, indent=2, ensure_ascii=False)+'\n```\n\n'
                'All optimizer updates, loss-bearing outputs, gradients, optimizer states, weights, validation predictions, '
                'test-step encodings/logits/probabilities and evaluation draws are saved under `details/`. '
                'All available-candidate selection uses varying pool sizes; best-of-N uses a fixed >=4-candidate cohort. '
                'Neither measures original-paper model quality.\n')
        state['stages'][-1].update(state='complete', ended_at_unix=time.time())
        state['state'] = 'complete'
        write_json(output/'pipeline.json', state)
        write_json(output/'artifacts.json', {str(p.relative_to(output)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifacts.json'})
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except Exception as exc:
        state.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'pipeline.json', state)
        raise
