"""Materialize the immutable official release as raw JSONL and individual JSON."""
from pathlib import Path
import shutil

from .assets import ASSETS, UPSTREAM, download
from .data import usable
from .io import problem_id, read_jsonl, sha256, write_json, write_jsonl

LABEL_ASSETS = ('phase1_train', 'phase1_test', 'phase2_train', 'phase2_test')


def materialize(output='datasets/prm800k'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'upstream_commit': UPSTREAM, 'format': 'one original JSONL record per JSON file; no relabeling', 'sources': {}}
    for name in LABEL_ASSETS:
        destination = output/'raw'/f'{name}.jsonl'
        cached = Path('.cache/data')/f'{name}.jsonl'
        if not destination.exists() and cached.exists() and sha256(cached) == ASSETS[name][1]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, destination)
        raw = download(name, output/'raw')
        print(f'Materializing {name}', flush=True)
        counts = {'records': 0, 'usable_records': 0}
        def records():
            for index, row in enumerate(read_jsonl(raw)):
                relative = Path('records')/name/f'{index//1000:03d}'/f'{index:07d}.json'
                target = output/relative
                write_json(target, row)
                counts['records'] += 1
                counts['usable_records'] += int(usable(row))
                yield {'record_index': index, 'path': str(relative), 'sha256': sha256(target),
                       'problem_id': problem_id(row['question']['problem']), 'usable': usable(row)}
        index_path = output/'indexes'/f'{name}.jsonl.gz'
        write_jsonl(index_path, records())
        receipt['sources'][name] = {**counts, 'raw_file': str(raw.relative_to(output)),
            'raw_sha256': sha256(raw), 'expected_sha256': ASSETS[name][1],
            'index_file': str(index_path.relative_to(output)), 'index_sha256': sha256(index_path)}
        write_json(output/'manifest.json', receipt)
    receipt['complete'] = True
    receipt['total_records'] = sum(s['records'] for s in receipt['sources'].values())
    write_json(output/'manifest.json', receipt)
    print(f'{receipt["total_records"]} individual JSON records saved in {output / "records"}', flush=True)
    return receipt
