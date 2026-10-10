#!/usr/bin/env python3
"""Read-only configuration checks for either role on any supported GB10 host."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from deployment_checks import display_kv_checks, validate_model_mount
from settings import load_settings
from fabric_selectors import parse_selectors
from model_source import HUB_MOUNT, cache_snapshot, compose_command


def main():
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(root)
    manifest = json.loads((root / 'manifests/source.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    for name in ('files/entrypoint.sh', 'scripts/cluster.sh', 'scripts/legacy.sh', 'start.sh', 'stop.sh', 'restart.sh', 'status.sh', 'tail-log.sh', 'check.sh', 'sync-repo.sh', 'verify.sh'):
        subprocess.run(['bash', '-n', str(root / name)], check=True)
    rendered = subprocess.check_output(compose_command(root, settings) +
        ['config', '--format', 'json'], text=True)
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
    parse_selectors(env['FABRIC_SUBNETS'].split())
    subprocess.run(['docker', 'image', 'inspect', s['image']], check=True, stdout=subprocess.DEVNULL)
    for volume in s['volumes']:
        assert Path(volume['source']).exists(), volume['source']
        if volume['target'].startswith(('/usr/local/lib/', '/opt/')) or volume['target'] == '/deployment':
            assert Path(volume['source']).resolve().is_relative_to(root), 'Source patch is outside this deployment'
        if volume['target'].startswith('/models/') or volume['target'] in (HUB_MOUNT, '/weight-snapshot-seed'):
            assert volume['read_only'], 'Model and seed weights must be read-only'
    if env.get('MODEL_ID'):
        assert env.get('HF_HUB_CACHE') == HUB_MOUNT
        assert env.get('HF_HUB_OFFLINE') == '1' and env.get('TRANSFORMERS_OFFLINE') == '1'
        host = next(Path(v['source']) for v in s['volumes'] if v['target'] == HUB_MOUNT)
        models = []
        for prefix in ('MODEL', 'DRAFT'):
            local = cache_snapshot(host, env.get(prefix + '_ID'), env.get(prefix + '_REVISION'))
            models.append((prefix, host, HUB_MOUNT, str(Path(HUB_MOUNT) / local.relative_to(host))))
    else:
        assert not env.get('DRAFT_ID'), 'DRAFT_ID requires HF MODEL_ID mode'
        models = [(key, next(Path(v['source']) for v in s['volumes'] if v['target'] == target),
                   target, env[key]) for key, target in
                  (('MODEL_DIR', '/models/glm-5.3-flash-nvfp4'), ('DFLASH_MODEL', '/models/glm-5.3-flash-dflash2'))]
    for key, host, target, path in models:
        config = validate_model_mount(host, target, path, tokenizer=key in ('MODEL', 'MODEL_DIR'), label=key)
        if key in ('MODEL', 'MODEL_DIR'):
            limit = config.get('text_config', config).get('max_position_embeddings', 1048576)
            assert int(env['MAX_MODEL_LEN']) <= limit
    errors, advisories = display_kv_checks(env)
    if errors:
        raise ValueError('\n'.join(errors))
    for advisory in advisories:
        print(f'WARN: {advisory}', file=sys.stderr)
    if subprocess.run(['systemctl', 'is-active', '--quiet', 'earlyoom'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        print('WARN: earlyoom is active; it can terminate a cold checkpoint load. Inspect its journal and see docs/DEPLOYMENT.md#host-oom-daemons.', file=sys.stderr)
    print(f'{env["ROLE"]}: configuration, model files/symlinks, mounts, image and source integrity PASS (review any host warnings); recorded validation is in manifests/validation-summary.json')


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f'Configuration check failed: {error}')
