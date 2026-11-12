"""PRM endpoint cross-entropy and ORM all-solution-token cross-entropy."""
from dataclasses import asdict
from contextlib import nullcontext
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .data import step_examples, candidates
from .io import sha256, write_json
from .model import ModelConfig, RewardModel, batch_tokens, device_for, encode, save_model, seed_all


def prepare(rows, kind, context, include_metadata=False):
    if include_metadata:
        from .annotation import trajectory_id
        for row_index, row in enumerate(rows):
            identity = trajectory_id(row)
            if kind == 'prm':
                for index, example in enumerate(step_examples([row])):
                    tokens, start, truncated = encode(example.problem, example.prefix + [example.step], context)
                    yield tokens, start, example.rating+1, truncated, dict(asdict(example), trajectory_id=identity,
                        raw_partition_row_index=row_index, projected_example_index=index)
            else:
                for encoded in prepare([row], kind, context):
                    yield (*encoded, {'trajectory_id': identity, 'raw_partition_row_index': row_index,
                        'problem': row['question']['problem'], 'steps': row['question']['pre_generated_steps'],
                        'label_origin': 'teacher_outcome' if kind == 'orm-teacher' else 'answer_grader'})
        return
    if kind == 'prm':
        for example in step_examples(rows):
            tokens, start, truncated = encode(example.problem, example.prefix + [example.step], context)
            yield tokens, start, example.rating + 1, truncated
    elif kind in ('orm', 'orm-teacher'):
        pool = candidates(rows) if kind == 'orm' else ({'problem': row['question']['problem'], 'steps': row['question']['pre_generated_steps'], 'is_correct': row['teacher_outcome']} for row in rows)
        for sample in pool:
            tokens, start, truncated = encode(sample['problem'], sample['steps'], context)
            if len(tokens) > start:
                yield tokens, start, int(sample['is_correct']), truncated
    else:
        raise ValueError('kind must be prm, orm or orm-teacher')


def loss_and_metrics(model, batch, kind, device, details=None):
    tokens, lengths = batch_tokens([x[0] for x in batch], device)
    logits = model(tokens)
    if details is not None:
        from .tracing import forward_details
        details.update(forward_details(batch, logits, kind))
    labels = torch.tensor([x[2] for x in batch], device=device)
    final = logits[torch.arange(len(batch), device=device), lengths - 1]
    if kind == 'prm':
        loss = F.cross_entropy(final, labels)
    else:
        targets = torch.full_like(tokens, -100)
        for i, (_, start, target, _) in enumerate(batch):
            targets[i, start:lengths[i]] = target
        loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
    accuracy = (final.argmax(-1) == labels).float().sum().item()
    return loss, accuracy, final


def train(train_rows, validation_rows, output, *, kind='prm', device='auto', seed=42,
          epochs=None, batch_size=32, learning_rate=0.001, config=None, max_examples=12000, trace_dir=None):
    overlap = {r['question']['problem'] for r in train_rows} & {r['question']['problem'] for r in validation_rows}
    if overlap:
        raise ValueError(f'Train/validation leakage: {len(overlap)} shared problems')
    config = config or ModelConfig()
    return _fit(prepare(train_rows, kind, config.context, trace_dir is not None), prepare(validation_rows, kind, config.context, trace_dir is not None),
                output, kind=kind, device=device, seed=seed, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, config=config, max_examples=max_examples, trace_dir=trace_dir)


def train_steps(training_examples, validation_examples, output, *, device='auto', seed=42,
                epochs=2, batch_size=32, learning_rate=0.001, config=None, metadata=None, trace_dir=None):
    """Train on an explicit, frozen list of labeled prefixes without resampling.

    Used by protocol comparisons so the exact input examples and equal budgets
    are auditable. Synthetic post-error labels remain explicitly identified.
    """
    if not training_examples or not validation_examples:
        raise ValueError('Nonempty explicit training and validation examples required')
    overlap = {x.problem for x in training_examples} & {x.problem for x in validation_examples}
    if overlap:
        raise ValueError(f'Train/validation leakage: {len(overlap)} shared problems')
    config = config or ModelConfig()
    def encoded(examples):
        for example in examples:
            if type(example.rating) is not int or example.rating not in (-1, 0, 1):
                raise ValueError('Step rating must be -1, 0 or 1')
            tokens, start, truncated = encode(example.problem, example.prefix + [example.step], config.context)
            encoded_example = (tokens, start, example.rating+1, truncated)
            yield (*encoded_example, asdict(example)) if trace_dir is not None else encoded_example
    return _fit(encoded(training_examples), encoded(validation_examples), output, kind='prm',
                device=device, seed=seed, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, config=config,
                max_examples=max(len(training_examples), len(validation_examples)), extra_metadata=metadata, trace_dir=trace_dir)


def _fit(training_stream, validation_stream, output, *, kind, device, seed, epochs,
         batch_size, learning_rate, config, max_examples, extra_metadata=None, trace_dir=None):
    epochs = epochs if epochs is not None else (2 if kind == 'prm' else 1)
    if epochs < 1 or batch_size < 1 or max_examples < 1 or learning_rate <= 0:
        raise ValueError('epochs, batch size, learning rate and max examples must be positive')
    seed_all(seed)
    device = device_for(device)
    config = config or ModelConfig()
    config = ModelConfig(**dict(asdict(config), classes=3 if kind == 'prm' else 2))
    model = RewardModel(config).to(device)
    import hashlib
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        if not name.startswith('head.'):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    initial_backbone_hash = digest.hexdigest()
    rng = random.Random(seed)
    from .tracing import TrainingTrace, cpu_tree
    trace = TrainingTrace(trace_dir, kind) if trace_dir is not None else None
    def bounded(stream, limit, split):
        selected = []
        with trace.stream(f'{split}_examples.jsonl.gz') if trace else nullcontext(None) as append:
            for i, original in enumerate(stream):
                example = original[:4]
                if trace:
                    append(trace.input_row(original, split, i))
                chosen = i if i < limit else rng.randrange(i+1)
                if chosen < limit:
                    if trace:
                        if chosen < len(selected):
                            trace.example_ids.pop(id(selected[chosen]))
                        trace.example_ids[id(example)] = f'{split}:{i}'
                    if chosen == len(selected):
                        selected.append(example)
                    else:
                        selected[chosen] = example
        return selected
    training = bounded(training_stream, max_examples, 'train')
    validation = bounded(validation_stream, max_examples, 'validation')
    if not training or not validation:
        raise ValueError(f'No usable {kind} examples. ORM requires complete pre-generated solutions.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    if trace:
        trace.json('selected_examples.json.gz', {'train': [trace.example_ids[id(x)] for x in training],
                   'validation': [trace.example_ids[id(x)] for x in validation]})
        trace.capture_state(model, optimizer, rng, epoch=0, update=0)
    history = []
    update = 0
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        rng.shuffle(training)
        total_loss = 0.0
        for offset in range(0, len(training), batch_size):
            batch = training[offset:offset+batch_size]
            optimizer.zero_grad(set_to_none=True)
            details = {} if trace else None
            loss, _, _ = loss_and_metrics(model, batch, kind, device, details)
            loss.backward()
            before_gradients = {n: cpu_tree(p.grad) for n, p in model.named_parameters() if p.grad is not None} if trace else None
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update += 1
            if trace:
                record = trace.batch_record(batch, details, epoch=epoch+1, update=update, phase='train',
                                           loss=loss.item(), gradient_norm=gradient_norm.item())
                trace.json(f'batches/update_{update:06d}.json.gz', record)
                trace.capture_state(model, optimizer, rng, epoch=epoch+1, update=update,
                                    before_gradients=before_gradients, batch=record)
            if update % 100 == 0:
                print(f'{kind.upper()} epoch {epoch+1}: update {update}, examples {offset+len(batch)}/{len(training)}', flush=True)
            total_loss += loss.item() * len(batch)
        model.eval()
        val_loss, correct = 0.0, 0.0
        confusion = torch.zeros((config.classes, config.classes), dtype=torch.int64)
        with torch.inference_mode(), trace.stream(f'validation/epoch_{epoch+1:03d}.jsonl.gz') if trace else nullcontext(None) as append_validation:
            for offset in range(0, len(validation), batch_size):
                batch = validation[offset:offset+batch_size]
                details = {} if trace else None
                loss, count, logits = loss_and_metrics(model, batch, kind, device, details)
                if trace:
                    append_validation(trace.batch_record(batch, details, epoch=epoch+1, update=update,
                                                        phase='validation', loss=loss.item()))
                val_loss += loss.item() * len(batch)
                correct += count
                for target, prediction in zip([x[2] for x in batch], logits.argmax(-1).cpu().tolist()):
                    confusion[target, prediction] += 1
        support = confusion.sum(1).float()
        recall = confusion.diag().float() / support.clamp(min=1)
        precision = confusion.diag().float() / confusion.sum(0).float().clamp(min=1)
        f1 = 2*precision*recall/(precision+recall).clamp(min=1e-12)
        record = {'balanced_accuracy': recall[support>0].mean().item(),
                  'macro_f1': f1[support>0].mean().item(),
                  'majority_class_baseline': support.max().item()/support.sum().item(),
                  'epoch': epoch+1, 'train_loss': total_loss/len(training),
                  'validation_loss': val_loss/len(validation), 'validation_accuracy': correct/len(validation),
                  'confusion_matrix': confusion.tolist()}
        history.append(record)
        if trace:
            write_json(trace.path/f'epoch_{epoch+1:03d}.json', dict(record, final_update=update))
        print(f'{kind.upper()} epoch {epoch+1}/{epochs}: loss={record["train_loss"]:.4f}, validation={record["validation_accuracy"]:.3f}', flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'supervised_tokens_per_epoch': sum(1 if kind == 'prm' else len(x[0])-x[1] for x in training),
                'optimizer_steps': epochs * ((len(training)+batch_size-1)//batch_size),
                'initial_backbone_sha256': initial_backbone_hash,
                'kind': kind, 'config': asdict(config), 'device': str(device), 'seed': seed,
                'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
                'parameters': sum(p.numel() for p in model.parameters()), 'seconds': time.perf_counter()-start,
                'training_examples': len(training), 'validation_examples': len(validation),
                'truncated_training_examples': sum(x[3] for x in training),
                'truncated_validation_examples': sum(x[3] for x in validation),
                'train_validation_overlap': 0, 'history': history,
                'scope': 'Small randomly initialized causal model; not a reproduction of GPT-4 quality.'}
    if extra_metadata:
        metadata.update(extra_metadata)
    if trace:
        trace.finish(optimizer_updates=update, training_examples=len(training), validation_examples=len(validation),
                     epochs=epochs, checkpoint_count=update+1, loss_kind=kind,
                     scope='All input projections, loss-bearing logits, raw/clipped gradients, optimizer states and weights at every update; no hidden-layer activation dump.')
        metadata['trace'] = {'directory': str(trace.path), 'optimizer_updates': update,
                             'checkpoint_count': update+1, 'manifest_sha256': sha256(trace.path/'artifacts.json')}
    save_model(output / f'{kind}.pt', model, metadata)
    write_json(output / f'{kind}_training.json', metadata)
    return model.eval(), metadata
