#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=${GOOSE_OCC_ROOT:-/path/to/GOOSE-OCC}
LOG="$ROOT/work_dir/goose_dataset_checksum_audit.log"
OUTPUT="$ROOT/work_dir/goose_dataset_checksum_audit.json"
PYTHON=python

mkdir -p "$ROOT/work_dir"
started=$(date --iso-8601=seconds)
if (cd "$DATA_ROOT" && sha256sum -c checksums.sha256) > "$LOG" 2>&1; then
  status=pass
  exit_code=0
else
  status=fail
  exit_code=1
fi
finished=$(date --iso-8601=seconds)

"$PYTHON" - "$DATA_ROOT" "$LOG" "$OUTPUT" "$status" \
  "$started" "$finished" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data_root, log_path, output_path, status, started, finished = sys.argv[1:]
data_root = Path(data_root)
log_path = Path(log_path)
manifest = data_root / 'checksums.sha256'

digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
payload = {
    'status': status,
    'dataset_root': str(data_root),
    'dataset_version': (data_root / 'VERSION').read_text().strip(),
    'checksum_manifest': str(manifest),
    'checksum_manifest_sha256': digest,
    'started_at': started,
    'finished_at': finished,
    'checked_ok_lines': sum(line.endswith(': OK') for line in lines),
    'failed_lines': [line for line in lines if 'FAILED' in line],
    'log': str(log_path),
}
Path(output_path).write_text(
    json.dumps(payload, indent=2) + '\n', encoding='utf-8')
print(json.dumps(payload, indent=2))
PY

exit "$exit_code"

