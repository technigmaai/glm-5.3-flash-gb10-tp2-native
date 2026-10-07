#!/usr/bin/env python3
"""Synchronize tracked source only, with both deployment folders offline."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from settings import load_settings

# Shared local/remote guard; never edit a source tree mounted by a live container.
GUARD = '''
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(sys.argv[1]).resolve()
ids=subprocess.check_output(['docker','ps','-q'],text=True).split()
containers=json.loads(subprocess.check_output(['docker','inspect',*ids])) if ids else []
for c in containers:
 for m in c.get('Mounts',[]):
  source=Path(m['Source']).resolve()
  if source.is_relative_to(root) or root.is_relative_to(source):
   raise SystemExit('Refusing sync: deployment source is mounted by running container '+c['Name'])
env=root/'.env'
print(json.dumps({'env_sha256':hashlib.sha256(env.read_bytes()).hexdigest() if env.exists() else None}))
'''


def tracked_source(names):
    result = []
    for name in names:
        p = Path(name)
        if (p.is_absolute() or '..' in p.parts or p.name.startswith('.env') and p.name != '.env.example'
                or any(part in {'.git', 'site', 'logs', 'experiments', 'backups', 'reports', 'models', 'cache', '__pycache__'} for part in p.parts)):
            continue
        result.append(name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--approved', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(root)
    if settings['ROLE'] != 'head':
        parser.error('Run source sync from the head')
    peer = settings['PEER_SSH']; destination = settings['PEER_DEPLOY_DIR']
    if not peer or peer.startswith('-') or any(c.isspace() for c in peer):
        parser.error('PEER_SSH must be a valid SSH destination')
    if not Path(destination).is_absolute():
        parser.error('PEER_DEPLOY_DIR must be absolute')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', peer]
    remote = shlex.join(['python3', '-c', GUARD, destination])
    subprocess.run(['python3', '-c', GUARD, str(root)], check=True, stdout=subprocess.DEVNULL)
    before = json.loads(subprocess.check_output(ssh + [remote], text=True))
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    names = tracked_source(n for n in names if n)
    with tempfile.NamedTemporaryFile() as selection:
        selection.write(('\0'.join(names) + '\0').encode()); selection.flush()
        command = ['rsync', '-az', '--checksum', '--itemize-changes', '--protect-args', '--from0',
                   '--files-from=' + selection.name]
        if args.dry_run:
            command.append('--dry-run')
        # No --delete; remote Git metadata and node-local files remain untouched.
        subprocess.run(command + [str(root) + '/', peer + ':' + destination + '/'], check=True)
    after = json.loads(subprocess.check_output(ssh + [remote], text=True))
    if before != after:
        raise SystemExit('Worker .env changed during synchronization; investigate before starting')
    print('Source sync dry-run complete.' if args.dry_run else 'Tracked source synced; worker .env preserved. Commit/check out the same revision on both nodes.')


if __name__ == '__main__':
    main()
