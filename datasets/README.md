# Local official data

Run `prm800k materialize-data` to download the four official PRM800K annotation
files and write all 101,599 records as individual JSON objects under
`prm800k/records/<asset>/<shard>/<record>.json` in this directory.

The original files are under `prm800k/raw/`. Git tracks the source manifest,
per-record indexes and validation receipt; raw files and expanded copies remain
on the local machine. No annotation fields are removed during extraction.
Training applies usability/step-label filters separately.

See [complete local workflow](../docs/FULL_TRACE.md).
