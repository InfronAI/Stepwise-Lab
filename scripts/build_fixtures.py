"""Rebuild offline fixtures from hash-verified official assets. Run from repo root."""
from collections import Counter
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prm800k.assets import download, ASSETS, UPSTREAM
from prm800k.data import usable
from prm800k.io import read_jsonl, sha256, write_json, write_jsonl


def main():
    output = Path('prm800k/fixtures')
    rng = random.Random(42)
    sources = {}
    training = []
    for name, limit in [('phase1_train', 128), ('phase2_train', 640)]:
        path = download(name)
        sources[name] = {'sha256': sha256(path), 'expected_sha256': ASSETS[name][1]}
        # Reservoir sampling, independent of correctness or length.
        kept = []
        seen = 0
        for row in read_jsonl(path):
            if not usable(row):
                continue
            seen += 1
            if len(kept) < limit:
                kept.append(row)
            else:
                replace = rng.randrange(seen)
                if replace < limit:
                    kept[replace] = row
        training.extend(kept)
        sources[name]['eligible_rows'] = seen
    path = download('phase2_test')
    all_test = [r for r in read_jsonl(path) if usable(r)]
    counts = Counter(r['question']['problem'] for r in all_test)
    eligible = sorted(p for p, count in counts.items() if count >= 4)
    rng.shuffle(eligible)
    selected = set(eligible[:32])
    # Preserve all samples for each selected problem; cap evaluation N to min count.
    testing = [r for r in all_test if r['question']['problem'] in selected]
    train_problems = {r['question']['problem'] for r in training}
    assert not train_problems.intersection(selected)
    sources['phase2_test'] = {'sha256': sha256(path), 'expected_sha256': ASSETS['phase2_test'][1]}
    write_jsonl(output/'train.jsonl.gz', training)
    write_jsonl(output/'test.jsonl.gz', testing)
    write_json(output/'manifest.json', {'upstream_commit': UPSTREAM, 'seed': 42, 'sources': sources,
        'selection': 'train: reservoir of usable rows (128 phase1 + 640 phase2); test: 32 random problems with >=4 usable rows, all their candidates',
        'train_rows': len(training), 'test_rows': len(testing), 'test_problems': len(selected),
        'train_test_overlap': 0, 'minimum_test_candidates': min(Counter(r['question']['problem'] for r in testing).values()),
        'files': {p.name: sha256(p) for p in [output/'train.jsonl.gz',output/'test.jsonl.gz']}})
    print('Fixtures:', len(training), 'train,', len(testing), 'test rows')

if __name__ == '__main__':
    main()
