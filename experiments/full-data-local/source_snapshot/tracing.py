"""Lossless, disk-backed records of algorithm steps; no random draws or reruns."""
from contextlib import contextmanager
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import random

from .io import sha256, write_json


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def write_tensor_archive(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as stream:
            torch.save(payload, stream)
    os.replace(temporary, path)


def load_tensor_archive(path):
    import torch
    with gzip.open(path, 'rb') as stream:
        return torch.load(stream, map_location='cpu', weights_only=True)


class TraceStore:
    def __init__(self, path, kind):
        self.path = Path(path)
        if self.path.exists() and any(self.path.iterdir()):
            raise ValueError(f'Trace directory is not empty: {path}')
        self.path.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.counts = {}
        write_json(self.path/'status.json', {'state': 'running', 'kind': kind})

    @contextmanager
    def stream(self, name):
        path = self.path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.tmp')
        count = 0
        with temporary.open('wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as stream:
                def append(record):
                    nonlocal count
                    stream.write((json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n').encode())
                    count += 1
                yield append
        os.replace(temporary, path)
        self.counts[name] = count

    def json(self, name, record):
        with self.stream(name) as append:
            append(record)

    def finish(self, **details):
        write_json(self.path/'status.json', {'state': 'complete', 'kind': self.kind,
                   'record_counts': self.counts, **details})
        files = {str(p.relative_to(self.path)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                 for p in sorted(self.path.rglob('*')) if p.is_file() and p.name != 'artifacts.json'}
        write_json(self.path/'artifacts.json', files)


class TrainingTrace(TraceStore):
    def __init__(self, path, kind):
        super().__init__(path, f'{kind}_training')
        self.example_ids = {}
        self.updates = 0

    def capture_state(self, model, optimizer, rng, *, epoch, update, before_gradients=None, batch=None):
        import torch
        state = {'config': asdict(model.config), 'state_dict': cpu_tree(model.state_dict()),
                 'optimizer_state_dict': cpu_tree(optimizer.state_dict()), 'epoch': epoch, 'update': update,
                 'torch_rng_state': torch.get_rng_state(), 'python_rng_state': random.getstate(),
                 'shuffle_rng_state': rng.getstate()}
        if next(model.parameters()).device.type == 'mps':
            state['mps_rng_state'] = torch.mps.get_rng_state().cpu()
        if before_gradients is not None:
            state['gradients_before_clip'] = before_gradients
            state['gradients_after_clip'] = {n: cpu_tree(p.grad) for n, p in model.named_parameters() if p.grad is not None}
        if batch is not None:
            state['batch'] = batch
        write_tensor_archive(self.path/'states'/f'update_{update:06d}.pt.gz', state)

    def input_row(self, item, split, index):
        tokens, start, label, truncated = item[:4]
        return {'id': f'{split}:{index}', 'token_ids': tokens, 'solution_start': start,
                'class_target': label, 'truncated_solution': bool(truncated),
                'text_and_provenance': item[4] if len(item)>4 else None}

    def batch_record(self, batch, details, *, epoch, update, phase, loss, gradient_norm=None):
        return {'epoch': epoch, 'update': update, 'phase': phase,
                'example_ids': [self.example_ids[id(x)] for x in batch],
                'batch_loss': loss, 'gradient_norm_before_clip': gradient_norm,
                'clip_max_norm': 1.0 if gradient_norm is not None else None,
                **details}


def forward_details(batch, logits, kind):
    """Persist every loss-bearing logit, including every supervised ORM token."""
    import torch
    endpoint_logits, supervised_logits, targets, losses = [], [], [], []
    for i, item in enumerate(batch):
        tokens, start, label, _ = item[:4]
        endpoint = logits[i, len(tokens)-1].detach().cpu()
        selected = logits[i, len(tokens)-1:len(tokens)] if kind == 'prm' else logits[i, start:len(tokens)]
        selected = selected.detach().cpu()
        labels = torch.full((len(selected),), label, dtype=torch.long)
        losses.append(torch.nn.functional.cross_entropy(selected, labels, reduction='none').tolist())
        endpoint_logits.append(endpoint.tolist())
        supervised_logits.append(selected.tolist())
        targets.append(labels.tolist())
    return {'endpoint_logits': endpoint_logits,
            'endpoint_probabilities': torch.tensor(endpoint_logits).softmax(-1).tolist(),
            'supervised_logits': supervised_logits, 'supervised_targets': targets,
            'token_cross_entropies_recomputed_on_cpu': losses}
