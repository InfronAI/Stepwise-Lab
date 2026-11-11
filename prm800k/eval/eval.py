"""Upstream-compatible entry point, now using a compact cached score index."""
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prm800k.cli import main


def legacy_main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    flags = {arg.split('=')[0] for arg in argv}
    if not flags.intersection(('--input', '--database', '-h', '--help')):
        from prm800k.evaluation import build_index
        database = Path('.cache/scored.sqlite')
        if not database.exists():
            print('Indexing the official public score release. This downloads about 2.16 GB.', flush=True)
            build_index('https://openaipublic.blob.core.windows.net/process-supervision/scored-test-samples.jsonl', database)
        argv.extend(['--database', str(database)])
        defaults = {'--population': '1860', '--ns': '10,25,50,75,100,200,300,400,500,750,1000,1250,1500,1860',
                    '--trials': '400', '--output': 'experiments/runs/legacy-evaluation'}
        for flag, value in defaults.items():
            if flag not in flags:
                argv.extend([flag, value])
    main(['evaluate', *argv])


if __name__ == '__main__':
    legacy_main()
