"""Pinned public assets. No Azure credentials or Git LFS installation required."""
from pathlib import Path
import hashlib
import os
import time
import urllib.request

from .io import sha256, write_json

UPSTREAM = '7ecc794703b2877f63226f2477a49b34f9b25163'
BASE = f'https://media.githubusercontent.com/media/openai/prm800k/{UPSTREAM}/prm800k'
ASSETS = {
    'phase1_train': ('data/phase1_train.jsonl', 'e9da6a73f827ffb9a8c0dc644c541d34ed76b3d4d1e4896ff5f7b37ddf5ae34d', 7900236),
    'phase1_test': ('data/phase1_test.jsonl', 'f4b3bc5b095e45c816453dc4d748b755c680d61d55f9895d929a335b487c727d', 829105),
    'phase2_train': ('data/phase2_train.jsonl', '1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089', 456135365),
    'phase2_test': ('data/phase2_test.jsonl', '6b172efa884ac8341a946dd82e06947c135b7254109fb3f7aa907c715d98aaad', 12240719),
    'math_test': ('math_splits/test.jsonl', '35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132', 446564),
    'math_train': ('math_splits/train.jsonl', '90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076', 10896985),
    'scored': ('https://openaipublic.blob.core.windows.net/process-supervision/scored-test-samples.jsonl', None, 2159374091),
}


def download(name, directory='.cache/data'):
    relative, expected, size = ASSETS[name]
    url = relative if relative.startswith('https://') else f'{BASE}/{relative}'
    path = Path(directory) / f'{name}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size:
        actual = sha256(path)
        if expected is None or actual == expected:
            return path
    part = path.with_suffix('.partial')
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=90) as response, part.open('wb') as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(chunk)
                    digest.update(chunk)
            actual = digest.hexdigest()
            if part.stat().st_size != size or (expected and actual != expected):
                raise ValueError(f'Integrity check failed for {name}; partial file retained at {part}')
            os.replace(part, path)
            write_json(path.with_suffix('.source.json'), {'url': url, 'sha256': actual, 'bytes': size, 'upstream_commit': UPSTREAM, 'verification': 'upstream LFS SHA-256' if expected else 'size checked; locally computed SHA-256'})
            return path
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
