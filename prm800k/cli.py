"""One CLI for download, audit, train, generate, score and evaluate."""
import argparse
import json
from pathlib import Path
import sys

from .io import read_jsonl, write_json, write_jsonl


def parser():
    root = argparse.ArgumentParser(prog='prm800k', description=__doc__)
    sub = root.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor', help='Inspect native Python and optional MPS support')
    doctor.add_argument('--output')
    download = sub.add_parser('download', help='Download hash-verified upstream assets')
    from .assets import ASSETS
    download.add_argument('assets', nargs='+', choices=list(ASSETS))
    download.add_argument('--directory', default='.cache/data')
    audit = sub.add_parser('audit', help='Stream annotation statistics')
    audit.add_argument('paths', nargs='+')
    audit.add_argument('--output', default='experiments/runs/audit.json')
    demo = sub.add_parser('demo', help='Run the complete offline training/evaluation experiment')
    demo.add_argument('--output', default='experiments/runs/demo')
    demo.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    demo.add_argument('--seed', type=int, default=42)
    demo.add_argument('--epochs', type=int, default=2)
    demo.add_argument('--max-examples', type=int, default=6000)
    demo.add_argument('--context', type=int, default=256)
    train = sub.add_parser('train', help='Train a causal PRM or ORM checkpoint')
    train.add_argument('--train', nargs='+', required=True)
    train.add_argument('--validation', nargs='+')
    train.add_argument('--kind', choices=['prm', 'orm', 'orm-teacher'], default='prm')
    train.add_argument('--output', required=True)
    train.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--epochs', type=int)
    train.add_argument('--context', type=int, default=384)
    train.add_argument('--width', type=int, default=64)
    train.add_argument('--layers', type=int, default=2)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--max-examples', type=int, default=12000)
    score = sub.add_parser('score', help='Predict PRM/ORM scores for candidate JSONL')
    score.add_argument('--input', required=True)
    score.add_argument('--prm', required=True)
    score.add_argument('--orm')
    score.add_argument('--output', required=True)
    score.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    index = sub.add_parser('index', help='Stream released scores into a compact SQLite database')
    index.add_argument('--input', default='https://openaipublic.blob.core.windows.net/process-supervision/scored-test-samples.jsonl')
    index.add_argument('--output', default='.cache/scored.sqlite')
    index.add_argument('--limit', type=int)
    evaluate = sub.add_parser('evaluate', help='Evaluate local candidates or a full scored-sample index')
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument('--input')
    source.add_argument('--database')
    evaluate.add_argument('--method', default='prm,orm,majority', help='Comma-separated methods')
    evaluate.add_argument('--ns', default='1,2,4')
    evaluate.add_argument('--population', type=int, help='Total attempted generations per problem, including missing samples; 1860 for official replay')
    evaluate.add_argument('--trials', type=int, default=100)
    evaluate.add_argument('--seed', type=int, default=42)
    evaluate.add_argument('--output', required=True)
    active = sub.add_parser('active-select', help='Select convincing wrong-answer TRAINING candidates')
    active.add_argument('--input', required=True)
    active.add_argument('--output', required=True)
    active.add_argument('--budget', type=int, required=True)
    active.add_argument('--global-budget', action='store_true')
    synthetic = sub.add_parser('synthetic-labels', help='Make first-error labels from teacher probabilities')
    synthetic.add_argument('--input', required=True)
    synthetic.add_argument('--output', required=True)
    synthetic.add_argument('--threshold', type=float, default=0.2)
    generate = sub.add_parser('generate', help='Sample a local open model (optional transformers dependency)')
    generate.add_argument('--input', required=True, help='Problem JSONL with problem and answer fields')
    generate.add_argument('--output', required=True)
    generate.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    generate.add_argument('--revision')
    generate.add_argument('--adapter')
    generate.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    generate.add_argument('--samples', type=int, default=4)
    generate.add_argument('--max-new-tokens', type=int, default=256)
    generate.add_argument('--seed', type=int, default=42)
    generate.add_argument('--temperature', type=float, default=1.0)
    generate.add_argument('--top-p', type=float, default=1.0)
    generate.add_argument('--greedy', action='store_true')
    generate.add_argument('--local-files-only', action='store_true')
    ablate = sub.add_parser('ablate', help='Matched synthetic supervision and active-acquisition experiments')
    ablate.add_argument('--teacher', required=True)
    ablate.add_argument('--output', required=True)
    ablate.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    ablate.add_argument('--seeds', default='42,43,44')
    ablate.add_argument('--pool-limit', type=int, default=160)
    tune = sub.add_parser('finetune', help='Local LoRA generator SFT or supplied math-corpus adaptation')
    tune.add_argument('--input', required=True)
    tune.add_argument('--output', required=True)
    tune.add_argument('--mode', choices=['sft','math'], default='sft')
    tune.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    tune.add_argument('--revision')
    tune.add_argument('--device', choices=['auto','mps','cpu'], default='auto')
    tune.add_argument('--epochs', type=int, default=1)
    tune.add_argument('--context', type=int, default=256)
    tune.add_argument('--max-examples', type=int, default=32)
    protocols = sub.add_parser('annotation-ablate', help='Paired phase-1 branches / phase-2 first-error experiments')
    protocols.add_argument('--output', required=True)
    protocols.add_argument('--device', choices=['auto','mps','cpu'], default='auto')
    protocols.add_argument('--seeds', default='42,43,44')
    protocols.add_argument('--split-seed', type=int, default=42)
    protocols.add_argument('--budget', type=int, default=1024)
    protocols.add_argument('--epochs', type=int, default=2)
    protocols.add_argument('--context', type=int, default=256)
    protocols.add_argument('--batch-size', type=int, default=32)
    protocols.add_argument('--teacher', help='PRM checkpoint with adjacent inputs.json split provenance')
    protocols.add_argument('--teacher-pool-limit', type=int, default=96)
    protocols.add_argument('--threshold', type=float, default=0.2)
    protocols.add_argument('--families', default='phase1_branches,phase2_error_label,synthetic_horizon')
    protocols.add_argument('--train-input')
    protocols.add_argument('--test-input')
    replay = sub.add_parser('annotation-replay', help='Replay annotation state transitions from released JSONL')
    replay.add_argument('--input', required=True)
    replay.add_argument('--output', required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        dispatch(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc


def dispatch(args):
    command = args.command
    if command == 'doctor':
        from .experiment import environment
        result = environment()
        try:
            import torch
            result.update(mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available())
        except ImportError:
            result['training'] = 'Install with pip install -e ".[train]"'
        print(json.dumps(result, indent=2))
        if args.output:
            write_json(args.output, result)
    elif command == 'download':
        from .assets import download
        for name in args.assets:
            print(download(name, args.directory))
    elif command == 'audit':
        from .data import audit
        result = audit(args.paths)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif command == 'demo':
        from .experiment import run_demo
        run_demo(args.output, args.device, args.seed, args.epochs, args.max_examples, args.context)
    elif command == 'train':
        from .data import AnnotationSource
        from .model import ModelConfig
        from .training import train
        rows = AnnotationSource(args.train, None if args.validation else 'train', args.seed)
        validation = AnnotationSource(args.validation or args.train, None if args.validation else 'validation', args.seed)
        from .io import sha256
        write_json(Path(args.output)/'inputs.json', {'files': {p: sha256(p) for p in args.train+(args.validation or [])}, 'split': 'explicit' if args.validation else 'SHA-256 problem split, 20% validation', 'seed': args.seed})
        train(rows, validation, args.output, kind=args.kind, device=args.device, seed=args.seed,
              epochs=args.epochs, config=ModelConfig(context=args.context, width=args.width, layers=args.layers),
              batch_size=args.batch_size, max_examples=args.max_examples)
    elif command == 'score':
        from .model import load_model
        from .scoring import score_candidates
        prm, _ = load_model(args.prm, args.device)
        orm = load_model(args.orm, args.device)[0] if args.orm else None
        write_jsonl(args.output, score_candidates(read_jsonl(args.input), prm, orm))
        from .io import sha256
        write_json(Path(args.output).with_suffix('.scoring.json'), {'input_sha256': sha256(args.input), 'prm_sha256': sha256(args.prm), 'orm_sha256': sha256(args.orm) if args.orm else None, 'device': args.device})
    elif command == 'index':
        from .evaluation import build_index
        print(f'{build_index(args.input, args.output, args.limit):,} rows indexed')
    elif command == 'evaluate':
        from .evaluation import evaluate, indexed_groups, group_samples
        from .report import render_report
        samples = list(read_jsonl(args.input)) if args.input else []
        groups = indexed_groups(args.database) if args.database else sorted(group_samples(samples).items())
        result = evaluate(groups, [int(x) for x in args.ns.split(',')], population=args.population,
                          methods=args.method.split(','), trials=args.trials, seed=args.seed)
        print(render_report(args.output, result, samples, title='PRM800K · Precomputed-score evaluation' if args.database else 'PRM800K · Candidate evaluation'))
        from .io import sha256
        source = args.database or args.input
        write_json(Path(args.output)/'inputs.json', {'source': str(source), 'sha256': sha256(source), 'population': args.population, 'methods': args.method, 'ns': args.ns})
        for method, metrics in result['methods'].items():
            print(method, metrics.get('exact_mean', metrics['mean']))
    elif command == 'active-select':
        from .supervision import active_select
        write_jsonl(args.output, active_select(read_jsonl(args.input), args.budget, per_problem=not args.global_budget))
    elif command == 'synthetic-labels':
        from .supervision import synthetic_rows
        write_jsonl(args.output, synthetic_rows(read_jsonl(args.input), args.threshold))
    elif command == 'ablate':
        from .ablations import run_ablation
        run_ablation(args.teacher, args.output, args.device, tuple(int(x) for x in args.seeds.split(',')), args.pool_limit)
    elif command == 'annotation-replay':
        from .annotation import replay_annotation
        from .data import usable
        write_jsonl(args.output, (replay_annotation(r) for r in read_jsonl(args.input) if usable(r)))
    elif command == 'annotation-ablate':
        from .protocol_experiments import run_protocol_experiments
        run_protocol_experiments(args.output, device=args.device, seeds=tuple(int(x) for x in args.seeds.split(',')),
            split_seed=args.split_seed, budget=args.budget, epochs=args.epochs, context=args.context,
            batch_size=args.batch_size, teacher_path=args.teacher, teacher_pool_limit=args.teacher_pool_limit,
            threshold=args.threshold, families=args.families.split(','), train_path=args.train_input, test_path=args.test_input)
    elif command == 'finetune':
        from .finetuning import finetune
        metadata = finetune(read_jsonl(args.input), args.output, mode=args.mode, model_name=args.model, revision=args.revision, device=args.device, epochs=args.epochs, context=args.context, max_examples=args.max_examples)
        from .io import sha256
        metadata['input_sha256'] = sha256(args.input)
        write_json(Path(args.output)/'training.json', metadata)
    elif command == 'generate':
        from .generation import generate
        samples, meta = generate(list(read_jsonl(args.input)), model_name=args.model, revision=args.revision,
                                 device=args.device, samples_per_problem=args.samples,
                                 max_new_tokens=args.max_new_tokens, seed=args.seed, local_files_only=args.local_files_only, adapter=args.adapter, temperature=args.temperature, top_p=args.top_p, greedy=args.greedy)
        write_jsonl(args.output, samples)
        from .io import sha256
        meta['input_sha256'] = sha256(args.input)
        write_json(Path(args.output).with_suffix('.generation.json'), meta)
