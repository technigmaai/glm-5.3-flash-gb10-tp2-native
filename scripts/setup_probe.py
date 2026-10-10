#!/usr/bin/env python3
"""Standard-library host inventory and bounded TCP probes; sent over SSH by setup."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys


def run(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=15)
        return p.returncode == 0, p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)


def hf_command(home):
    """SSH command shells may omit ~/.local/bin even after a user CLI install."""
    found = shutil.which('hf')
    if found:
        return found
    local = Path(home) / '.local/bin/hf'
    return str(local) if local.is_file() and os.access(local, os.X_OK) else None


def source_hash(root):
    manifest = root / 'manifests/source.json'
    if not manifest.is_file():
        raise ValueError('Missing native checkout at ' + str(root) +
                         '; clone the same repository revision on both nodes or correct --worker-dir')
    data = json.loads(manifest.read_text())
    for name, expected in data['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Source path escapes checkout: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Source manifest mismatch: ' + name)
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def writable_checkout(root, expected):
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Expected an existing real checkout directory: ' + str(root))
    if os.path.lexists(root / '.env'):
        raise ValueError('Existing .env is protected; use manual configuration for an existing deployment')
    if source_hash(root) != expected:
        raise ValueError('Checkout changed since the setup plan; regenerate the plan')
    ok, ids = run(['docker', 'ps', '-q'])
    if not ok:
        raise ValueError('Cannot inspect running containers; check Docker permissions: ' + ids)
    if ids:
        ok, details = run(['docker', 'inspect', *ids.split()])
        if not ok:
            raise ValueError('Cannot inspect Docker bind mounts: ' + details)
        for container in json.loads(details):
            for mount in container.get('Mounts', []):
                if not mount.get('Source') or mount.get('Type', 'bind') != 'bind':
                    continue
                path = Path(mount['Source']).resolve()
                if path == root.resolve() or path.is_relative_to(root.resolve()) or root.resolve().is_relative_to(path):
                    raise ValueError('Checkout overlaps a running container mount: ' + container.get('Name', '?'))


def inventory(root):
    home = Path.home()
    ok, raw = run(['ip', '-j', '-4', 'addr'])
    if not ok:
        raise ValueError('Cannot enumerate IPv4 interfaces: ' + raw)
    addresses = json.loads(raw)
    rdma = set()
    for device in Path('/sys/class/infiniband').glob('*'):
        rdma.update(p.name for p in (device / 'device/net').glob('*'))
    for link in addresses:
        link['rdma'] = link['ifname'] in rdma
    checks = {}
    for name, command in [('docker', ['docker', 'info']), ('compose', ['docker', 'compose', 'version']),
                          ('gpu_driver', ['nvidia-smi', '-L'])]:
        ok, detail = run(command)
        checks[name] = {'ok': ok, 'detail': detail[:500]}
    cli = hf_command(home)
    checks['tools'] = {name: bool(shutil.which(name)) for name in ('bash', 'git', 'flock', 'curl', 'jq')}
    checks['tools']['hf'] = bool(cli)
    checks['arm64'] = platform.machine() in ('aarch64', 'arm64')
    try:
        state = subprocess.run(['systemctl', 'is-active', 'display-manager'], capture_output=True, text=True, timeout=5)
        checks['headless'] = state.stdout.strip() in ('inactive', 'failed', 'unknown') and not state.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        checks['headless'] = False
    mode = Path('/sys/module/nvidia_drm/parameters/modeset')
    try:
        modeset = mode.read_text().strip()
    except PermissionError:
        ok, modeset = run(['sudo', '-n', 'cat', str(mode)])
        if not ok:
            modeset = 'unknown'
    except OSError:
        modeset = 'missing'
    checks['drm_modeset'] = None if modeset == 'unknown' else modeset in ('Y', '1')
    card = Path('/dev/dri/card0')
    checks['drm_card'] = card.exists()
    base = os.environ.get('HF_HOME') or str(Path(os.environ.get('XDG_CACHE_HOME', str(home / '.cache'))) / 'huggingface')
    hub = os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE') or str(Path(base) / 'hub')
    # Do not dump the environment or any HF credentials into reports.
    firewall = 'absent'
    if shutil.which('ufw'):
        ok, text = run(['sudo', '-n', 'ufw', 'status']) if os.geteuid() else run(['ufw', 'status'])
        firewall = ('inactive' if 'Status: inactive' in text else 'active' if 'Status: active' in text else 'unknown') if ok else 'unknown'
    return dict(root=str(root), home=str(home), hub=str(Path(hub).expanduser().absolute()),
                addresses=addresses, drm_gid=str(card.stat().st_gid) if card.exists() else None,
                source_sha256=source_hash(root), existing_env=os.path.lexists(root / '.env'),
                ssh_connection=os.environ.get('SSH_CONNECTION', ''), checks=checks, ufw=firewall, hf_command=cli)


def apply(payload):
    root = Path(payload['root'])
    writable_checkout(root, payload['source_sha256'])
    for directory in payload['directories']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    fd = os.open(root / '.env', os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w') as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(payload['env'])
    return {'created': str(root / '.env')}


def tcp_server(data):
    """One ephemeral listener, one authenticated peer, hard timeout, no permanent service."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.settimeout(data.get('timeout', 15))
        server.bind((data['address'], 0))
        server.listen(1)
        print(json.dumps({'port': server.getsockname()[1]}), flush=True)
        conn, peer = server.accept()
        with conn:
            conn.settimeout(data.get('timeout', 15))
            if peer[0] != data['peer']:
                raise ValueError('Unexpected probe peer: ' + peer[0])
            expected = data['token'].encode()
            received = b''
            while len(received) < len(expected):
                part = conn.recv(len(expected) - len(received))
                if not part:
                    raise ValueError('Incomplete probe challenge')
                received += part
            if received != expected:
                raise ValueError('Incorrect probe challenge')
            conn.sendall(expected)
    return {'ok': True}


def tcp_client(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(data.get('timeout', 10))
        client.bind((data['source'], 0))
        client.connect((data['address'], data['port']))
        expected = data['token'].encode()
        client.sendall(expected)
        received = b''
        while len(received) < len(expected):
            part = client.recv(len(expected) - len(received))
            if not part:
                raise ValueError('Incomplete probe response')
            received += part
        if received != expected:
            raise ValueError('Incorrect probe response')
    return {'ok': True}


def main():
    mode = sys.argv[1]
    payload = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.load(sys.stdin)
    if mode == 'inventory':
        result = inventory(Path(payload['root']))
    elif mode == 'guard':
        writable_checkout(Path(payload['root']), payload['source_sha256'])
        result = {'ok': True}
    elif mode == 'apply':
        result = apply(payload)
    elif mode == 'server':
        result = tcp_server(payload)
    elif mode == 'client':
        result = tcp_client(payload)
    else:
        raise ValueError('Unknown mode: ' + mode)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
