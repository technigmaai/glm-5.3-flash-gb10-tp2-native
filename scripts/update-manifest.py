#!/usr/bin/env python3
"""Refresh tracked source hashes after intentional, reviewed source edits."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
p = root / 'manifests/source.json'
m = json.loads(p.read_text())
names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
m['files'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(names)
              if name and name != 'manifests/source.json' and (root / name).is_file()}
m['native_entrypoint_sha256'] = m['files']['files/entrypoint.sh']
p.write_text(json.dumps(m, indent=2) + '\n')
print('Tracked source manifest refreshed.')
