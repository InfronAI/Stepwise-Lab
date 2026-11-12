"""Paired annotation-protocol experiments with frozen data and equal budgets."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import json
from pathlib import Path
import random
import statistics
import time

from .annotation import (annotation_phase, example_dict, phase1_examples, phase2_examples,
                         replay_annotation, teacher_examples)
from .data import candidates, split_by_problem, usable
from .evaluation import evaluate, group_samples
from .experiment import FIXTURES, environment
from .io import problem_id, read_jsonl, sha256, write_json, write_jsonl
from .model import ModelConfig, device_for, load_model
from .scoring import score_candidates
from .training import train_steps


CONDITIONS = {
    'phase1_branches': ('all_candidates', 'chosen_path'),
    'phase2_error_label': ('include_first_error', 'mask_first_error'),
    'synthetic_horizon': ('teacher_first_error', 'teacher_full_trajectory'),
}


def example_stats(examples):
    return {'examples': len(examples), 'problems': len({e.problem for e in examples}),
            'trajectories': len({e.trajectory_id for e in examples}),
            'labels': {str(label): sum(e.rating == label for e in examples) for label in (-1, 0, 1)},
            'label_origins': dict(Counter(e.label_origin for e in examples)),
            'post_error_targets': sum(e.after_first_error for e in examples)}


def paired_subsets(left, right, budget, seed):
    """Equal per-problem quotas and total endpoint targets, no replacement.

    Source examples and class distributions may differ: that is the mechanism
    under study. Problems with no eligible targets in either arm are excluded
    from BOTH training arms and counted in the receipt.
    """
    if budget < 1:
        raise ValueError('budget must be positive')
    groups = []
    for examples in (left, right):
        grouped = defaultdict(list)
        for example in examples:
            grouped[example.problem].append(example)
        groups.append(grouped)
    common = sorted(set(groups[0]) & set(groups[1]))
    if not common:
        raise ValueError('No common eligible training problems')
    if budget < len(common):
        raise ValueError(f'Budget {budget} cannot cover {len(common)} common problems; increase --budget')
    capacities = {p: min(len(g[p]) for g in groups) for p in common}
    target = min(budget, sum(capacities.values()))
    quotas = dict.fromkeys(common, 1)
    rng = random.Random(seed)
    while sum(quotas.values()) < target:
        available = [p for p in common if quotas[p] < capacities[p]]
        rng.shuffle(available)
        for p in available[:target-sum(quotas.values())]:
            quotas[p] += 1
    chosen = []
    for grouped in groups:
        chooser = random.Random(seed)
        selected = []
        for problem in common:
            selected.extend(chooser.sample(grouped[problem], quotas[problem]))
        chosen.append(selected)
    receipt = {'requested_budget': budget, 'actual_budget': target,
               'shared_problems': len(common),
               'excluded_problems': [len(set(g)-set(common)) for g in groups],
               'quotas_by_problem_id': {problem_id(p): n for p, n in quotas.items()},
               'sampling': 'equal per-problem quotas; uniform without replacement within each arm'}
    return chosen[0], chosen[1], receipt


def _validate_teacher_split(metadata, inputs_path, training, validation, testing):
    """Refuse an unverifiable teacher/test overlap instead of relabeling test data."""
    if metadata.get('kind') != 'prm' or metadata['config']['classes'] != 3:
        raise ValueError('Synthetic horizon experiment requires a three-class PRM teacher')
    if not inputs_path.exists():
        raise ValueError('Teacher needs adjacent inputs.json with train/validation/test problem IDs')
    provenance = json.loads(inputs_path.read_text())
    actual_train = {problem_id(r['question']['problem']) for r in training}
    excluded = {problem_id(r['question']['problem']) for r in validation + testing}
    teacher_train = set(provenance['train_problem_ids'])
    if teacher_train & excluded or not teacher_train.issubset(actual_train):
        raise ValueError('Teacher split is incompatible with this experiment; use the original --split-seed or a matching teacher')
    return {'teacher_training_problem_ids': sorted(teacher_train), 'teacher_overlap_with_validation_or_test': 0}


def run_protocol_experiments(output, *, device='auto', seeds=(42, 43, 44), split_seed=42,
                             budget=1024, epochs=2, context=256, batch_size=32,
                             teacher_path=None, teacher_pool_limit=96, threshold=0.2,
                             families=None, train_path=None, test_path=None, trace=False):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct, nonempty training seeds')
    if min(budget, epochs, batch_size, teacher_pool_limit) < 1 or context < 32:
        raise ValueError('Positive budgets/epochs/batches and context >=32 required')
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0,1]')
    families = list(CONDITIONS) if families is None else list(families)
    if not families or len(set(families)) != len(families) or set(families)-set(CONDITIONS):
        raise ValueError('Choose distinct known experiment families')
    if 'synthetic_horizon' in families and not teacher_path:
        raise ValueError('Provide --teacher for synthetic_horizon, or request only human-label families')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory to preserve experiment evidence')
    device = device_for(device)
    train_path, test_path = Path(train_path or FIXTURES/'train.jsonl.gz'), Path(test_path or FIXTURES/'test.jsonl.gz')
    raw = list(read_jsonl(train_path))
    test_rows = list(read_jsonl(test_path))
    training, validation = split_by_problem(raw, seed=split_seed)
    split_sets = [{r['question']['problem'] for r in rows} for rows in (training, validation, test_rows)]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise ValueError('Train/validation/test problem overlap')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_root = Path(__file__).parent
    write_json(output/'source_hashes.json', {str(p.relative_to(source_root)): sha256(p)
               for p in sorted(source_root.rglob('*.py'))})
    write_json(output/'inputs.json', {'train_sha256': sha256(train_path), 'test_sha256': sha256(test_path),
               'split_seed': split_seed, 'train_problem_ids': sorted(map(problem_id, split_sets[0])),
               'validation_problem_ids': sorted(map(problem_id, split_sets[1])),
               'test_problem_ids': sorted(map(problem_id, split_sets[2])), 'overlap': 0})
    config = {'seeds': list(seeds), 'split_seed': split_seed, 'budget': budget, 'epochs': epochs,
              'batch_size': batch_size, 'model': asdict(ModelConfig(context=context)), 'families': families,
              'teacher_pool_limit': teacher_pool_limit, 'teacher_threshold': threshold,
              'trace_every_update': trace,
              'teacher_sha256': sha256(teacher_path) if teacher_path else None}
    write_json(output/'config.json', config)
    # Freeze exact source rows and replay every actual input trajectory, including
    # human replacements and phase2 terminal alternatives.
    for name, rows in [('train', training), ('validation', validation), ('test', test_rows)]:
        write_jsonl(output/f'{name}_trajectories.jsonl.gz', rows)
    traces = [replay_annotation(row) for row in raw + test_rows if usable(row)]
    write_jsonl(output/'annotation_traces.jsonl.gz', traces)
    write_json(output/'replay_audit.json', {
        'trajectories': len(traces), 'excluded_unusable_trajectories': len(raw)+len(test_rows)-len(traces),
        'phases': dict(Counter(t['phase'] for t in traces)),
        'human_replacement_events': sum(e['human_replacement'] for t in traces for e in t['events']),
        'phase1_branch_events': sum(e['candidate_count'] > 1 for t in traces if t['phase'] == 1 for e in t['events']),
        'phase2_first_error_stops': sum(t['phase'] == 2 and t['finish_reason'] == 'found_error' for t in traces),
        'unlabeled_tail_steps_preserved_without_human_labels': sum(t['unlabeled_tail_steps'] for t in traces if t['phase'] == 2)})
    test_candidates = list(candidates(test_rows))
    write_jsonl(output/'test_candidates.jsonl.gz', test_candidates)
    if not test_candidates:
        raise ValueError('No complete held-out candidates')
    smallest = min(map(len, group_samples(test_candidates).values()))
    ns = [n for n in (1, 2, 4) if n <= smallest]
    pools, validation_pools = {}, {}
    if 'phase1_branches' in families:
        tr = [r for r in training if annotation_phase(r) == 1]
        va = [r for r in validation if annotation_phase(r) == 1]
        pools['phase1_branches'] = (list(phase1_examples(tr)), list(phase1_examples(tr, 'chosen')))
        validation_pools['phase1_branches'] = list(phase1_examples(va))
    if 'phase2_error_label' in families:
        tr = [r for r in training if annotation_phase(r) == 2]
        va = [r for r in validation if annotation_phase(r) == 2]
        pools['phase2_error_label'] = (list(phase2_examples(tr)), list(phase2_examples(tr, False)))
        validation_pools['phase2_error_label'] = list(phase2_examples(va))
    if 'synthetic_horizon' in families:
        teacher, metadata = load_model(teacher_path, str(device))
        receipt = _validate_teacher_split(metadata, Path(teacher_path).parent/'inputs.json', training, validation, test_rows)
        receipt['teacher_sha256'] = sha256(teacher_path)
        write_json(output/'teacher_provenance.json', receipt)
        rng = random.Random(split_seed)
        def frozen_pool(rows, name):
            eligible = list(candidates(rows))
            selected = rng.sample(eligible, min(teacher_pool_limit, len(eligible)))
            scored = score_candidates(selected, teacher, trace_dir=output/'teacher_trace'/name if trace else None)
            for sample in scored:
                sample['teacher_sha256'] = config['teacher_sha256']
            return scored
        print('Scoring fixed training/validation candidates with the frozen local teacher', flush=True)
        synthetic_train, synthetic_val = frozen_pool(training, 'train'), frozen_pool(validation, 'validation')
        write_jsonl(output/'teacher_train_predictions.jsonl.gz', synthetic_train)
        write_jsonl(output/'teacher_validation_predictions.jsonl.gz', synthetic_val)
        pools['synthetic_horizon'] = (list(teacher_examples(synthetic_train, threshold=threshold)),
                                      list(teacher_examples(synthetic_train, stop_at_first_error=False, threshold=threshold)))
        validation_pools['synthetic_horizon'] = list(teacher_examples(synthetic_val, stop_at_first_error=False, threshold=threshold))
        del teacher
    results = []
    dataset_summary = {}
    for family in families:
        left, right = pools[family]
        validation_examples = validation_pools[family]
        destination = output/family
        destination.mkdir()
        write_jsonl(destination/'validation_examples.jsonl.gz', map(example_dict, validation_examples))
        validation_hash = sha256(destination/'validation_examples.jsonl.gz')
        dataset_summary[family] = {'available': {name: example_stats(pool) for name, pool in zip(CONDITIONS[family], (left, right))},
                                   'validation': example_stats(validation_examples), 'validation_sha256': validation_hash}
        for seed in seeds:
            arms = paired_subsets(left, right, budget, seed)
            pairing = arms[2]
            write_json(destination/f'pairing_seed{seed}.json', pairing)
            paired_metadata = []
            for condition, examples in zip(CONDITIONS[family], arms[:2]):
                run = destination/f'{condition}-seed{seed}'
                run.mkdir()
                write_jsonl(run/'training_examples.jsonl.gz', map(example_dict, examples))
                stats = example_stats(examples)
                write_json(run/'dataset.json', stats)
                provenance = {'family': family, 'condition': condition,
                              'training_examples_sha256': sha256(run/'training_examples.jsonl.gz'),
                              'validation_examples_sha256': validation_hash,
                              'label_origins': stats['label_origins'], 'post_error_targets': stats['post_error_targets']}
                print(f'{family}/{condition}, seed={seed}, targets={len(examples)}, problems={stats["problems"]}', flush=True)
                model, metadata = train_steps(examples, validation_examples, run, device=str(device), seed=seed,
                                              epochs=epochs, batch_size=batch_size, config=ModelConfig(context=context), metadata=provenance,
                                              trace_dir=run/'trace/training' if trace else None)
                paired_metadata.append(metadata)
                scored = score_candidates(test_candidates, model, trace_dir=run/'trace/inference' if trace else None)
                write_jsonl(run/'scored.jsonl.gz', scored)
                metrics = evaluate(sorted(group_samples(scored).items()), ns, methods=['prm'], trials=100, seed=split_seed,
                                   trace_dir=run/'trace/evaluation' if trace else None)
                write_json(run/'evaluation.json', metrics)
                results.append({'family': family, 'condition': condition, 'seed': seed,
                                'training_examples': len(examples), 'optimizer_steps': metadata['optimizer_steps'],
                                'training_label_counts': stats['labels'], 'post_error_targets': stats['post_error_targets'],
                                'validation_accuracy': metadata['history'][-1]['validation_accuracy'],
                                'validation_balanced_accuracy': metadata['history'][-1]['balanced_accuracy'],
                                'validation_macro_f1': metadata['history'][-1]['macro_f1'],
                                'initial_backbone_sha256': metadata['initial_backbone_sha256'],
                                'validation_examples_sha256': validation_hash,
                                'training_seconds': metadata['seconds'],
                                'exact_best_of_n': metrics['methods']['prm']['exact_mean']})
                del model
            for key in ('initial_backbone_sha256', 'optimizer_steps', 'supervised_tokens_per_epoch', 'validation_examples_sha256'):
                if paired_metadata[0][key] != paired_metadata[1][key]:
                    raise AssertionError(f'Unmatched protocol experiment: {key}')
    write_json(output/'datasets.json', dataset_summary)
    summary = summarize(results, ns, config)
    summary['test_set'] = {'problems': len(group_samples(test_candidates)), 'candidates': len(test_candidates)}
    summary['seconds'] = time.perf_counter()-started
    summary['environment'] = environment(str(device))
    write_json(output/'summary.json', summary)
    write_jsonl(output/'runs.jsonl.gz', results)
    write_reports(output, summary, dataset_summary)
    write_json(output/'artifacts.json', {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifacts.json'})
    print(f'Protocol experiments complete: {output / "ANALYSIS.zh-CN.md"}', flush=True)
    return summary


def summarize(results, ns, config):
    aggregated, paired = {}, {}
    for family in config['families']:
        aggregated[family] = {}
        for condition in CONDITIONS[family]:
            runs = [r for r in results if r['family'] == family and r['condition'] == condition]
            values = [r['exact_best_of_n'][-1] for r in runs]
            aggregated[family][condition] = {'seeds': [r['seed'] for r in runs],
                'best_of_n_mean': [statistics.mean(r['exact_best_of_n'][i] for r in runs) for i in range(len(ns))],
                'best_of_n_std_across_seeds': [statistics.stdev(r['exact_best_of_n'][i] for r in runs) if len(runs)>1 else None for i in range(len(ns))],
                'endpoint_best_mean': statistics.mean(values),
                'validation_balanced_accuracy_mean': statistics.mean(r['validation_balanced_accuracy'] for r in runs),
                'validation_macro_f1_mean': statistics.mean(r['validation_macro_f1'] for r in runs),
                'training_examples': runs[0]['training_examples']}
        left, right = CONDITIONS[family]
        by_condition = {condition: {r['seed']: r for r in results if r['family'] == family and r['condition'] == condition} for condition in (left, right)}
        deltas = [by_condition[left][s]['exact_best_of_n'][-1]-by_condition[right][s]['exact_best_of_n'][-1] for s in config['seeds']]
        paired[family] = {'comparison': f'{left} minus {right}', 'seeds': config['seeds'],
                          'differences': deltas, 'mean_difference': statistics.mean(deltas),
                          'sample_std_difference': statistics.stdev(deltas) if len(deltas)>1 else None}
    return {'ns': ns, 'config': config, 'aggregated': aggregated, 'paired_differences': paired,
            'runs': results, 'interpretation': 'Within-family controlled comparisons; synthetic post-error targets are not human labels. A small biased subset does not establish general superiority.'}


def write_reports(output, summary, datasets):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    families = summary['config']['families']
    seed_count = len(summary['config']['seeds'])
    test_set = summary['test_set']
    fig, axes = plt.subplots(1, len(families), figsize=(5*len(families), 4), squeeze=False, layout='constrained')
    for ax, family in zip(axes[0], families):
        for color, condition in zip(('#086bcb', '#d57737'), CONDITIONS[family]):
            metrics = summary['aggregated'][family][condition]
            deviations = [v or 0 for v in metrics['best_of_n_std_across_seeds']]
            ax.errorbar(summary['ns'], metrics['best_of_n_mean'], yerr=deviations, marker='o', capsize=3, label=condition, color=color)
        ax.set(title=family, xlabel='N', ylabel='Exact selection accuracy', ylim=(0, 1), xticks=summary['ns'])
        ax.legend(fontsize=8)
        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(alpha=.15)
    fig.savefig(output/'protocol_comparison.svg')
    fig.savefig(output/'protocol_comparison.png', dpi=160)
    plt.close(fig)
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['family', 'condition', 'seed', 'training_examples', 'optimizer_steps', 'validation_macro_f1', *[f'exact_best_of_{n}' for n in summary['ns']]])
        for run in summary['runs']:
            writer.writerow([run[k] for k in ('family', 'condition', 'seed', 'training_examples', 'optimizer_steps', 'validation_macro_f1')]+run['exact_best_of_n'])
    for language in ('en', 'zh'):
        zh = language == 'zh'
        lines = ['# 标注流程对照实验' if zh else '# Annotation protocol experiments', '',
                 '![Protocol comparison](protocol_comparison.svg)', '',
                 (f'每个条件使用相同模型结构、固定题目划分和 {seed_count} 个训练种子；同组内采用相同的题目配额、训练目标数量、优化步数和验证集。图中误差线为训练种子间的样本标准差，不是置信区间。'
                  if zh else 'Within each family, arms share problem quotas, endpoint-target budgets, optimizer steps, seeded model initialization and validation data. Error bars are sample standard deviations across training seeds, not confidence intervals.'), '',
                 f'| Family | Condition | Training targets | Best-of-{summary["ns"][-1]} (mean ± SD) | Validation macro-F1 |', '|---|---|---:|---:|---:|']
        for family in families:
            for condition, metrics in summary['aggregated'][family].items():
                sd = metrics['best_of_n_std_across_seeds'][-1]
                sd_text = f'{sd:.2%}' if sd is not None else 'not estimated'
                lines.append(f'| {family} | {condition} | {metrics["training_examples"]} | {metrics["endpoint_best_mean"]:.2%} ± {sd_text} | {metrics["validation_macro_f1_mean"]:.3f} |')
        lines.extend(['', '## Paired differences', '', '| Family | Left minus right | Mean difference |', '|---|---|---:|'])
        for family, result in summary['paired_differences'].items():
            lines.append(f'| {family} | {result["comparison"]} | {100*result["mean_difference"]:+.2f} percentage points |')
        lines.extend(['', '## Dataset diagnostics', '', '| Family | Condition | Available targets | Negative / neutral / positive | Post-error targets |', '|---|---|---:|---|---:|'])
        for family, data in datasets.items():
            for condition, stats in data['available'].items():
                lines.append(f'| {family} | {condition} | {stats["examples"]} | {stats["labels"]["-1"]} / {stats["labels"]["0"]} / {stats["labels"]["1"]} | {stats["post_error_targets"]} |')
        lines.extend(['', '## ' + ('解释与边界' if zh else 'Interpretation and limits'), ''])
        if zh:
            lines.extend([
                '- 第一组只比较第一阶段候选保留策略；两组都保留人工修复，并使用相同已选前缀。不能解释为在线交互标注成本实验。',
                '- 第二组只比较第二阶段原始轨迹的首错目标是否保留；两组均不训练首错之后的人工标签。首错处的替代候选在两组中都排除，避免引入分支数量差异。',
                '- 第三组才比较首错截断与全轨迹，其标签全部来自同一个本地教师。错误后的目标不是人工真值；弱教师误差和错误前缀都会影响结果。',
                '- 不同组的训练来源和验证目标不同，只能在同一组内部比较。训练按相同题目配额等量抽样，但类别比例、序列长度和可用标签多样性仍随策略变化。',
                f'- {test_set["problems"]} 道测试题、{test_set["candidates"]} 个候选；默认官方样本有主动学习偏置。小模型从零训练且上下文较短；{seed_count} 个种子不足以支持总体优劣或统计显著性结论。',
                '- 每个运行目录保存实际训练样本、模型、全部测试预测和指标；配对清单记录共同题目配额与排除数量，summary.json 保存逐种子差值。'])
        else:
            lines.extend([
                '- Phase 1 compares branch retention only; both arms retain human replacements and the same selected-history rule. It does not measure online labeling cost.',
                '- The human phase-2 comparison includes or masks the first erroneous original step. Neither arm invents human labels after that error. Terminal alternatives are excluded from both arms.',
                '- Only the synthetic family compares first-error truncation with full trajectories. Both arms use the same frozen local teacher; post-error targets are not human ground truth.',
                '- Compare arms within a family, not across families. Equal per-problem target quotas do not equalize class balance, sequence lengths or label diversity.',
                f'- The held-out subset has {test_set["problems"]} problems and {test_set["candidates"]} candidates; default official fixtures have active-selection bias. Models are small and trained from scratch with truncated context; {seed_count} seeds do not establish statistical significance or general superiority.',
                '- Exact training examples, checkpoint, all test predictions and metrics are retained per run. Pairing receipts record common quotas/exclusions and summary.json retains per-seed differences.'])
        (output/('ANALYSIS.zh-CN.md' if zh else 'ANALYSIS.md')).write_text('\n'.join(lines)+'\n')
