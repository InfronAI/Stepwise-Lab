"""Streaming JSONL, provenance and atomic artifact writes."""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.request


@contextlib.contextmanager
def open_text(path):
    path = str(path)
    if path.startswith('az://openaipublic/'):
        path = path.replace('az://openaipublic/', 'https://openaipublic.blob.core.windows.net/', 1)
    with contextlib.ExitStack() as stack:
        if path.startswith(('https://', 'http://')):
            raw = stack.enter_context(urllib.request.urlopen(path, timeout=90))
            if path.endswith('.gz'):
                raw = stack.enter_context(gzip.GzipFile(fileobj=raw))
            stream = stack.enter_context(io.TextIOWrapper(raw, encoding='utf-8'))
        else:
            opener = gzip.open if path.endswith('.gz') else open
            stream = stack.enter_context(opener(path, 'rt', encoding='utf-8'))
        yield stream


def read_jsonl(path):
    with open_text(path) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if line.startswith('version https://git-lfs.github.com/spec/'):
                raise ValueError(f'{path} is a Git LFS pointer. Run prm800k download or git lfs pull.')
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('expected a JSON object')
                yield row
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as raw:
        # Fixed gzip timestamp makes the bundled fixture byte-reproducible.
        with contextlib.ExitStack() as stack:
            output = stack.enter_context(gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0)) if path.suffix == '.gz' else raw
            for row in rows:
                output.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode())
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def problem_id(problem):
    return hashlib.sha256(problem.strip().encode()).hexdigest()[:20]
