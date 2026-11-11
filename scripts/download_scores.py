"""Bounded, resumable parallel download of the 2.16 GB public scored-sample release.

Every HTTP range is checked; the assembled file must match the Azure Content-MD5.
"""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import fcntl
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prm800k.io import write_json

URL='https://openaipublic.blob.core.windows.net/process-supervision/scored-test-samples.jsonl'
SIZE=2159374091
MD5='sAdWxw8ZVCLHhV2u0oIgXQ=='


def main():
    root=Path('.cache/data')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'scored.download.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('Another scored download is running') from exc
    target=root/'scored.jsonl'
    if target.exists() and target.stat().st_size==SIZE:
        cached_md5 = hashlib.md5()
        with target.open('rb') as stream:
            while data := stream.read(1024*1024):
                cached_md5.update(data)
        if base64.b64encode(cached_md5.digest()).decode() == MD5:
            print('Existing file verified:',target)
            return
        raise ValueError('Existing scored.jsonl failed Content-MD5; remove the corrupt cache file before retrying')
    chunk=64*1024*1024
    def fetch(start):
        end=min(SIZE,start+chunk)-1
        path=root/f'scored.range-{start}'
        if path.exists() and path.stat().st_size==end-start+1:
            return path
        for attempt in range(3):
            try:
                req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as response, path.with_name(path.name+'.partial').open('wb') as out:
                    if response.status != 206 or response.headers['Content-Range'] != f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('Server did not honor the requested range')
                    while data:=response.read(1024*1024):
                        out.write(data)
                part=path.with_name(path.name+'.partial')
                if part.stat().st_size!=end-start+1:
                    raise ValueError('Incomplete range')
                part.replace(path)
                return path
            except OSError:
                if attempt==2:
                    raise
                time.sleep(attempt+1)
    starts=list(range(0,SIZE,chunk))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures=[executor.submit(fetch,start) for start in starts]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            print(f'{i}/{len(starts)} ranges downloaded',flush=True)
    md5=hashlib.md5()
    digest=hashlib.sha256()
    with target.with_suffix('.partial').open('wb') as out:
        for start in starts:
            with (root/f'scored.range-{start}').open('rb') as stream:
                while data:=stream.read(1024*1024):
                    out.write(data)
                    md5.update(data)
                    digest.update(data)
    if base64.b64encode(md5.digest()).decode()!=MD5:
        raise ValueError('Azure Content-MD5 mismatch')
    target.with_suffix('.partial').replace(target)
    write_json(target.with_suffix('.source.json'),{'url':URL,'bytes':SIZE,'sha256':digest.hexdigest(),
               'content_md5':MD5,'verification':'Azure Content-MD5 and byte length verified'})
    for start in starts:
        (root/f'scored.range-{start}').unlink()
    print('Verified:',target,flush=True)

if __name__=='__main__':
    main()
