"""Validate recorded experiment evidence, local documentation links and demo media."""
import json
from pathlib import Path
import re
import subprocess

from prm800k.io import read_jsonl, sha256, write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    checks={}
    for name in ['m1-local','cpu-smoke']:
        directory=ROOT/'experiments'/name
        inputs=json.loads((directory/'inputs.json').read_text())
        train=set(inputs['train_problem_ids'])
        validation=set(inputs['validation_problem_ids'])
        test=set(inputs['test_problem_ids'])
        assert not train & validation and not train & test and not validation & test
        samples=list(read_jsonl(directory/'scored.jsonl.gz'))
        metrics=json.loads((directory/'evaluation.json').read_text())
        assert len({r['problem'] for r in samples})==metrics['problems']==32
        checks[name]={'samples':len(samples),'problems':metrics['problems'],'split_overlap':0}
    for filename in ['README.md','README.zh-CN.md']:
        text=(ROOT/filename).read_text()
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https://','http://','#')):
                continue
            assert (ROOT/target.split('#')[0]).exists(), (filename,target)
        checks[filename]='local links exist'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size','-show_entries','stream=codec_name,width,height','-of','json',str(ROOT/'docs/media/demo.mp4')]))
    assert float(probe['format']['duration'])>30
    assert probe['streams'][0]['codec_name']=='h264'
    checks['demo_video']=probe
    for directory in (ROOT/'experiments').glob('*'):
        if directory.is_dir() and directory.name!='runs':
            write_json(directory/'artifacts.json', {str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='artifacts.json'})
    checks['source_files_sha256'] = {str(p.relative_to(ROOT)):sha256(p) for directory in ['prm800k','scripts','tests'] for p in sorted((ROOT/directory).rglob('*.py'))}
    write_json(ROOT/'experiments/validation.json',checks)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    main()
