#!/usr/bin/env python3
"""Copy required source assets to the isolated folder; do not operate containers."""
import ast
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True, help='Pinned Kindling source checkout')
parser.add_argument('--display-patch-dir', type=Path, required=True, help='Matching display-memory patch source')
args = parser.parse_args()
origin = args.source_root.resolve()
display = args.display_patch_dir.resolve()
for destination_name, relative in json.loads((root / 'manifests/copy-assets.json').read_text()).items():
    source = display / Path(relative).name if relative.startswith("display-kv/") else origin / relative
    destination = root / destination_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
# Keep allocator provenance and source alongside the exact mounted Python file.
for source in display.iterdir():
    if source.is_file() and source.suffix != '.pyc' and source.name != 'Dockerfile':
        if source.resolve() != (root / 'files/display-kv' / source.name).resolve():
            shutil.copy2(source, root / 'files/display-kv' / source.name)

# Snapshot processing and keys remain unchanged. Prefer this deployment's own
# cache; optionally restore existing exact-key snapshots through a read-only bind.
# New or changed snapshots are written only to the isolated native cache.
p = root / 'files/overlays/snapshot/weight_snapshot.py'
s = p.read_text()
anchor = '        return cls(os.path.join(base, name), key)'
assert s.count(anchor) == 1
s = s.replace(anchor, '        return cls(_native_snapshot_path(base, name), key)')
helper = '''

def _native_snapshot_path(base, name):
    local = os.path.join(base, name)
    seed = os.environ.get("VLLM_WEIGHT_SNAPSHOT_SEED_DIR", "")
    if not seed or os.environ.get("VLLM_WEIGHT_SNAPSHOT_NORMAL") == "1":
        return local
    if os.path.isfile(os.path.join(local, "COMPLETE")):
        return local
    candidate = os.path.join(seed, name)
    if os.path.isfile(os.path.join(candidate, "COMPLETE")):
        return candidate
    return local
'''
p.write_text(s + helper)
ast.parse(p.read_text())
# Test cache isolation using the real generated helper without importing torch.
tree = ast.parse(helper)
scope = {'os': os}
exec(compile(tree, '<snapshot-selection>', 'exec'), scope)
select = scope['_native_snapshot_path']
previous = {k: os.environ.get(k) for k in ('VLLM_WEIGHT_SNAPSHOT_SEED_DIR', 'VLLM_WEIGHT_SNAPSHOT_NORMAL')}
try:
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary) / 'native'
        seed = Path(temporary) / 'seed'
        name = 'exact-key'
        os.environ['VLLM_WEIGHT_SNAPSHOT_SEED_DIR'] = str(seed)
        os.environ.pop('VLLM_WEIGHT_SNAPSHOT_NORMAL', None)
        assert select(str(cache), name) == str(cache / name)
        (seed / name).mkdir(parents=True)
        (seed / name / 'COMPLETE').touch()
        assert select(str(cache), name) == str(seed / name)
        os.environ['VLLM_WEIGHT_SNAPSHOT_NORMAL'] = '1'
        assert select(str(cache), name) == str(cache / name)
        os.environ.pop('VLLM_WEIGHT_SNAPSHOT_NORMAL')
        (cache / name).mkdir(parents=True)
        (cache / name / 'COMPLETE').touch()
        assert select(str(cache), name) == str(cache / name)
finally:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

# Hash only tracked source, so private backups and node settings stay private.
subprocess.run([sys.executable, str(root / 'scripts/update-manifest.py')], check=True)
print('Independent source assets copied; snapshot isolation cases PASS; tracked manifest updated')
