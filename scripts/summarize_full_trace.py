"""Build compact, shareable views of the complete local intermediate records."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np

from prm800k.io import read_jsonl, sha256, write_json, write_jsonl


def summarize(root):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    training = json.loads((root/'prm_training.json').read_text())
    cohort = json.loads((root/'evaluation_best_of_n.json').read_text())
    trace = root/'details/training'
    metrics = []
    for path in sorted((trace/'batches').glob('*.json.gz')):
        batch = next(read_jsonl(path))
        target = np.array([x[-1] for x in batch['supervised_targets']])
        prediction = np.array(batch['endpoint_logits']).argmax(-1)
        metrics.append({'update': batch['update'], 'epoch': batch['epoch'], 'examples': len(target),
            'batch_loss': batch['batch_loss'], 'batch_accuracy': float(np.mean(target==prediction)),
            'gradient_norm_before_clip': batch['gradient_norm_before_clip']})
    with (root/'training_steps.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(metrics)
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4), layout='constrained')
    losses = np.array([r['batch_loss'] for r in metrics])
    axes[0].plot(np.arange(1,len(losses)+1),losses, alpha=.22, linewidth=.5, color='#2277bb',label='Every update')
    window = min(100, len(losses))
    axes[0].plot(np.arange(window,len(losses)+1),np.convolve(losses,np.ones(window)/window,mode='valid'),
                 color='#224466',label=f'{window}-update mean')
    axes[0].set(title='Full-data local PRM training',xlabel='Optimizer update',ylabel='Cross-entropy')
    axes[0].legend(fontsize=8)
    confusion = np.array(training['history'][-1]['confusion_matrix'])
    axes[1].imshow(confusion/confusion.sum(1,keepdims=True).clip(min=1),vmin=0,vmax=1,cmap='Blues')
    for (i,j),value in np.ndenumerate(confusion):
        axes[1].text(j,i,f'{value:,}',ha='center',va='center',fontsize=9,
                     color='white' if value/max(1,confusion[i].sum())>.65 else 'black')
    axes[1].set(title='Validation: counts, row-normalized color',xticks=[0,1,2],xticklabels=['−1','0','+1'],
                yticks=[0,1,2],yticklabels=['−1','0','+1'],xlabel='Predicted',ylabel='Human target')
    if cohort['metrics']:
        evaluation = cohort['metrics']
        axes[2].plot(evaluation['ns'],evaluation['methods']['prm']['exact_mean'],marker='o')
        axes[2].set(xticks=evaluation['ns'],ylim=(0,1),xlabel='N',ylabel='Exact selection accuracy',
                    title=f'Fixed test cohort: {cohort["eligible_problems"]} problems')
    fig.savefig(root/'training_and_inference.png',dpi=160)
    fig.savefig(root/'training_and_inference.svg')
    plt.close(fig)
    preview = root/'preview'
    preview.mkdir(exist_ok=True)
    needed = set()
    for update in (0,1,training['optimizer_steps']):
        source = trace/'states'/f'update_{update:06d}.pt.gz'
        shutil.copyfile(source, preview/source.name)
        if update:
            batch_path = trace/'batches'/f'update_{update:06d}.json.gz'
            shutil.copyfile(batch_path, preview/batch_path.name)
            needed.update(next(read_jsonl(batch_path))['example_ids'])
    write_jsonl(preview/'referenced_training_examples.jsonl.gz',
                (r for r in read_jsonl(trace/'train_examples.jsonl.gz') if r['id'] in needed))
    shutil.copyfile(root/'details/inference/candidates/0000000.json.gz', preview/'first_test_candidate.json.gz')
    files = [p for p in (root/'details').rglob('*') if p.is_file()]
    trace_index = {'local_trace_files': len(files), 'local_trace_bytes': sum(p.stat().st_size for p in files),
        'checkpoint_count': training['optimizer_steps']+1, 'optimizer_updates': training['optimizer_steps'],
        'training_examples': training['training_examples'], 'validation_examples': training['validation_examples'],
        'test_candidates': summary['test_candidates'], 'test_step_predictions': summary['test_step_predictions'],
        'local_data': '../../datasets/prm800k', 'local_intermediates': 'details',
        'published_preview': 'preview', 'all_update_metrics': 'training_steps.csv',
        'scope': 'All algorithm-stage results and update states retained locally; hidden-layer activations are not dumped'}
    write_json(root/'trace_index.json', trace_index)
    final = training['history'][-1]
    for zh in (False,True):
        lines = ['# '+('完整本地 PRM 训练与推理' if zh else 'Full local PRM training and inference'), '',
                 '![Training and inference](training_and_inference.svg)', '',
                 '| Metric | Value |','|---|---:|',
                 f'| Training step labels | {training["training_examples"]:,} |',
                 f'| Validation step labels | {training["validation_examples"]:,} |',
                 f'| Optimizer updates / saved states | {training["optimizer_steps"]:,} / {training["optimizer_steps"]+1:,} |',
                 f'| Test problems / complete candidates | {summary["test_problems"]} / {summary["test_candidates"]:,} |',
                 f'| Test reasoning-step predictions | {summary["test_step_predictions"]:,} |',
                 f'| Validation accuracy / majority baseline | {final["validation_accuracy"]:.2%} / {final["majority_class_baseline"]:.2%} |',
                 f'| Validation balanced accuracy / macro-F1 | {final["balanced_accuracy"]:.3f} / {final["macro_f1"]:.3f} |',
                 f'| Top selection from all available candidates | {summary["available_candidate_top_selection_accuracy"]:.2%} |',
                 f'| Full intermediate storage | {trace_index["local_trace_bytes"]/1024**3:.2f} GiB |','']
        if cohort['metrics']:
            lines += [f'Fixed >=4-candidate cohort: {cohort["eligible_problems"]} of {cohort["all_problems"]} problems.', '',
                      '| N | Exact selection accuracy |','|---|---:|']
            lines += [f'| {n} | {value:.2%} |' for n,value in zip(cohort['metrics']['ns'],cohort['metrics']['methods']['prm']['exact_mean'])]
        lines += ['', (f'完整训练源经题目拆分后产生 {training["training_examples"]:,} 个训练标签和 {training["validation_examples"]:,} 个验证标签，未做数量截断。模型在本机训练，推理重新加载最终权重。所有逐步中间结果位于本地 `details/`；GitHub 保留清单、最终权重、逐更新 CSV、图表、校验记录及 `preview/` 示例。'
                       if zh else 'All eligible training-source labels are partitioned by problem without an example cap. Training executes locally and inference reloads the final checkpoint. Full intermediates are local under `details/`; GitHub retains manifests, final weights, every-update CSV, charts, verification and representative `preview/` files.'), '',
                  ('这是宽度 32、上下文 128 字节 token、从零训练一轮的小型 PRM；验证准确率需要结合多数类基线和 macro-F1 解读。全候选选优的每题候选数不同，不能与固定 N 的指标混为一谈；best-of-N 只在相同的至少四候选题目子集上比较。测试候选来自官方主动采样标注集，并不构成论文原始评分基准的复现。'
                   if zh else 'This is a randomly initialized width-32 PRM with 128 byte-token context and one epoch. Interpret validation accuracy alongside its majority-class baseline and macro-F1. All-available selection has varying pool sizes; fixed-N comparisons use the same >=4-candidate cohort. Official annotation candidates have active-selection bias and do not reproduce the paper score benchmark.'), '',
                  '[Detailed artifact map](../../docs/FULL_TRACE.md) · [Validation receipt](validation.json) · [Trace index](trace_index.json)', '']
        if final['validation_accuracy']-final['majority_class_baseline'] < .02:
            lines += [('验证准确率与多数类基线相差不到两个百分点，需结合较低的平衡准确率和 macro-F1 判断；不能把高占比正类上的预测当作可靠数学验证能力。'
                       if zh else 'Validation accuracy is within two percentage points of the majority-class baseline. Balanced accuracy and macro-F1 expose the weak discrimination; dominant-class predictions are not reliable mathematical verification.'), '']
        if cohort['metrics']:
            values = cohort['metrics']['methods']['prm']['exact_mean']
            if values[-1] <= values[0]:
                lines += [('在同一测试题子集上，增加到 best-of-4 没有超过 N=1 的精确期望准确率。本次验证了完整执行与记录链路，没有证明该小模型能改善候选选择。'
                           if zh else 'On the same test cohort, best-of-4 did not exceed the exact N=1 expectation. This run validates complete execution and traceability, not improved candidate selection.'), '']
        (root/('ANALYSIS.zh-CN.md' if zh else 'ANALYSIS.md')).write_text('\n'.join(lines))
    write_json(root/'artifacts.json', {str(p.relative_to(root)): {'sha256':sha256(p),'bytes':p.stat().st_size}
               for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'artifacts.json'})
    print(json.dumps(trace_index,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', default='experiments/full-data-local')
    summarize(parser.parse_args().run)
