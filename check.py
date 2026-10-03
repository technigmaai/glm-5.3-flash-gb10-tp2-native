#!/usr/bin/env python3
"""Read-only configuration checks for either role on any supported GB10 host."""
import hashlib
import json
from pathlib import Path
import subprocess
from settings import load_settings

root = Path(__file__).resolve().parent
settings = load_settings(root)
manifest = json.loads((root / 'manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
for name in ('entrypoint.sh', 'cluster.sh', 'legacy.sh'):
    subprocess.run(['bash', '-n', str(root / name)], check=True)
rendered = subprocess.check_output(['docker', 'compose', '--env-file', str(root / '.env'),
    '-p', settings.get('PROJECT_NAME', 'glm53-native'), '-f', str(root / 'compose.json'),
    'config', '--format', 'json'], text=True)
services = json.loads(rendered)['services']
assert list(services) == ['glm53'], 'Expected exactly one model service'
s = services['glm53']; env = s['environment']
assert env['ROLE'] in ('head', 'worker')
assert env['NODE_RANK'] == ('0' if env['ROLE'] == 'head' else '1')
assert int(env['TP']) == 2, 'This native recipe supports a two-node TP=2 GB10 pair'
assert 0 < int(env['MAX_MODEL_LEN']) <= 1048576
assert int(env['MAX_NUM_SEQS']) > 0 and int(env['KV_CACHE_MEMORY']) > 0
assert all(int(n) >= 0 for n in json.loads(env['LIMIT_MM']).values())
assert not any(k.startswith(('MENTAT_', 'RAY_')) for k in env)
subprocess.run(['docker', 'image', 'inspect', s['image']], check=True, stdout=subprocess.DEVNULL)
for volume in s['volumes']:
    assert Path(volume['source']).exists(), volume['source']
    if volume['target'].startswith(('/usr/local/lib/', '/opt/')) or volume['target'] == '/deployment':
        assert Path(volume['source']).resolve().is_relative_to(root), 'Source patch is outside this deployment'
    if volume['target'].startswith('/models/') or volume['target'] == '/weight-snapshot-seed':
        assert volume['read_only'], 'Model and seed weights must be read-only'
for key, target in (('MODEL_DIR', '/models/glm-5.3-flash-nvfp4'), ('DFLASH_MODEL', '/models/glm-5.3-flash-dflash2')):
    host = next(Path(v['source']) for v in s['volumes'] if v['target'] == target)
    config = json.loads((host / env[key].removeprefix(target + '/') / 'config.json').read_text())
    if key == 'MODEL_DIR':
        limit = config.get('text_config', config).get('max_position_embeddings', 1048576)
        assert int(env['MAX_MODEL_LEN']) <= limit
print(f'{env["ROLE"]}: configuration, mounts, image and source integrity PASS; runtime results are in validation.json')
