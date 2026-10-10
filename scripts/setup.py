#!/usr/bin/env python3
"""Plan or configure a fresh two-node deployment from the head, without starting it."""
import argparse
from datetime import datetime, timezone
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import select
import shlex
import subprocess
import sys

from configure import IMAGE, MODEL_ID, MODEL_REVISION, DRAFT_ID, DRAFT_REVISION, build_values, render
from fabric_selectors import select_fabric_addresses
import setup_probe

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).with_name('setup_probe.py')


def validate_worker(value):
    # No leading options, shell commands, control characters or IPv6 ambiguity.
    if not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*', value):
        raise ValueError('Worker must be an SSH hostname, IPv4 address or user@host; use SSH config for ports/jumps')
    return value


def remote_command(worker, mode, payload, code=None):
    validate_worker(worker)
    code = PROBE.read_text() if code is None else code
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', worker,
            shlex.join(['python3', '-c', code, mode, json.dumps(payload)])]


def request(worker, mode, payload):
    result = subprocess.run(remote_command(worker, mode, payload), capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise ValueError(f'Worker {mode} failed: {result.stderr.strip()}')
    return json.loads(result.stdout)


def endpoint(node, address):
    address = str(ipaddress.IPv4Address(address))
    matches = [(link, item) for link in node['addresses'] for item in link.get('addr_info', [])
               if item.get('family') == 'inet' and item['local'] == address]
    if len(matches) != 1:
        raise ValueError(f'{node["root"]}: expected one local interface for {address}, found {len(matches)}')
    link, item = matches[0]
    if link.get('operstate') not in ('UP', 'UNKNOWN'):
        raise ValueError('Interface is not up: ' + link['ifname'])
    # Reject unsafe interface names before generating UFW commands.
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', link['ifname']):
        raise ValueError('Unsupported interface name: ' + link['ifname'])
    return dict(address=address, interface=link['ifname'], rdma=link.get('rdma', False),
                mtu=link.get('mtu', 0), network=str(ipaddress.IPv4Interface(address + '/' + str(item['prefixlen'])).network))


def fabric_pairs(head, worker, selectors=None):
    pairs = []
    if selectors:
        for left, right in zip(select_fabric_addresses(head['addresses'], selectors),
                               select_fabric_addresses(worker['addresses'], selectors)):
            pairs.append((endpoint(head, left[1]), endpoint(worker, right[1])))
    else:
        for link in head['addresses']:
            if not link.get('rdma'):
                continue
            for item in link.get('addr_info', []):
                if item.get('family') != 'inet':
                    continue
                left = endpoint(head, item['local'])
                for peer in worker['addresses']:
                    if not peer.get('rdma'):
                        continue
                    for other in peer.get('addr_info', []):
                        if other.get('family') == 'inet':
                            right = endpoint(worker, other['local'])
                            if left['network'] == right['network'] and left['address'] != right['address']:
                                pairs.append((left, right))
    if len(pairs) != 2 or any(len({pair[side]['interface'] for pair in pairs}) != 2 for side in (0, 1)):
        raise ValueError('Expected exactly two distinct RoCE link pairs. Set --fabric-subnets "CIDR1 CIDR2" to resolve ambiguity')
    for pair in pairs:
        for node in pair:
            if not node['rdma'] or node['mtu'] < 9000:
                raise ValueError(f'{node["interface"]}: needs an RDMA device and MTU >= 9000; prepare fabric interfaces first')
        if pair[0]['address'] == pair[1]['address']:
            raise ValueError('Fabric peers cannot share an address')
    return sorted(pairs, key=lambda pair: int(ipaddress.IPv4Address(pair[0]['address'])))


def firewall_rules(pairs, side, client_cidr=None):
    rules = []
    for pair in pairs:
        local, peer = pair[side], pair[1 - side]
        rules.extend([
            ['ufw', 'allow', 'in', 'on', local['interface'], 'from', peer['address'], 'to', local['address']],
            ['ufw', 'allow', 'out', 'on', local['interface'], 'from', local['address'], 'to', peer['address']],
        ])
    if side == 0 and client_cidr:
        network = str(ipaddress.IPv4Network(client_cidr, strict=False))
        local = pairs[0][0]
        rules.append(['ufw', 'allow', 'in', 'on', local['interface'], 'proto', 'tcp', 'from', network,
                      'to', local['address'], 'port', '8000'])
    # Deduplicate management/fabric overlap; no global policies or open-to-world rules.
    return [list(rule) for rule in dict.fromkeys(tuple(rule) for rule in rules)]


def network_checks(worker, pairs):
    results = []
    for index, pair in enumerate(pairs):
        for reverse in (False, True):
            src, dest = (pair[1], pair[0]) if reverse else (pair[0], pair[1])
            token = secrets.token_hex(24)
            payload = dict(address=dest['address'], peer=src['address'], token=token, timeout=20)
            command = ([sys.executable, str(PROBE), 'server', json.dumps(payload)] if reverse
                       else remote_command(worker, 'server', payload))
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                ready, _, _ = select.select([process.stdout], [], [], 20)
                if not ready:
                    raise ValueError('Timed out creating the temporary listener')
                line = process.stdout.readline()
                if not line:
                    raise ValueError('Cannot bind listener: ' + process.stderr.read().strip())
                port = json.loads(line)['port']
                client = dict(address=dest['address'], source=src['address'], port=port, token=token, timeout=10)
                if reverse:
                    request(worker, 'client', client)
                else:
                    setup_probe.tcp_client(client)
                output, error = process.communicate(timeout=25)
                if process.returncode or not json.loads(output)['ok']:
                    raise ValueError(error.strip() or 'Listener did not confirm the probe')
                results.append(f'{src["address"]} -> {dest["address"]}: OK (temporary TCP port {port})')
            except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
                raise ValueError(f'Link {index}, {src["address"]} -> {dest["address"]}: {error}. Review peer firewall rules and routing') from error
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                process.stdout.close()
                process.stderr.close()
    return results


def plan(head, worker, target, head_ip=None, worker_ip=None, selectors=None, profile='c4', image=IMAGE, client_cidr=None, model_options=None):
    connection = worker['ssh_connection'].split()
    if not head_ip or not worker_ip:
        if len(connection) != 4:
            raise ValueError('Cannot discover SSH endpoints; supply --head-ip and --worker-ip')
        head_ip, worker_ip = head_ip or connection[0], worker_ip or connection[2]
    management = (endpoint(head, head_ip), endpoint(worker, worker_ip))
    if management[0]['address'] == management[1]['address']:
        raise ValueError('Head and worker must be different nodes')
    if head['source_sha256'] != worker['source_sha256']:
        raise ValueError('Source differs between nodes; clone the same repository revision on both')
    fabric = fabric_pairs(head, worker, selectors)
    pairs = [management, *fabric]
    values = []
    warnings = []
    for side, node in enumerate((head, worker)):
        if not node['drm_gid']:
            raise ValueError('Missing /dev/dri/card0 on ' + node['root'] + '; prepare the NVIDIA DRM driver first')
        values.append(build_values(root=node['root'], home=node['home'], hub=node['hub'],
                      role='head' if side == 0 else 'worker', head_host=management[0]['address'],
                      node_ip=management[side]['address'], fabric_subnets=' '.join(p[side]['address'] for p in fabric),
                      image=image, peer_ssh=target if side == 0 else '', peer_dir=worker['root'],
                      drm_gid=node['drm_gid'], profile=profile, **(model_options or {})))
        if node['existing_env']:
            warnings.append(('head' if side == 0 else 'worker') + ': existing .env is protected; --apply will refuse it')
        for key in ('docker', 'compose', 'gpu_driver'):
            if not node['checks'][key]['ok']:
                warnings.append(f'{side}: {key} unavailable: {node["checks"][key]["detail"]}')
        for key in ('arm64', 'headless', 'drm_modeset'):
            if node['checks'][key] is None:
                warnings.append(f'{side}: {key} unknown (root-only); confirm sudo cat /sys/module/nvidia_drm/parameters/modeset reports Y/1 before starting')
            elif not node['checks'][key]:
                warnings.append(f'{side}: host preparation needed: {key}')
        missing = [key for key, available in node['checks']['tools'].items() if not available]
        if missing:
            warnings.append(f'{side}: missing host tools: {", ".join(missing)}')
        if node['ufw'] in ('active', 'unknown'):
            warnings.append(f'{side}: UFW {node["ufw"]}; review generated rules before the network check')
    return dict(schema_version=1, worker=target, profile=profile, pairs=pairs, nodes=[head, worker],
                values=values, warnings=warnings, firewall=[firewall_rules(pairs, side, client_cidr) for side in (0, 1)])


def private_file(path, text):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        f.write(text)


def write_plan(data, path):
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    path.chmod(0o700)
    private_file(path / 'plan.json', json.dumps(data, indent=2) + '\n')
    for side, role in enumerate(('head', 'worker')):
        values = data['values'][side]
        cli = shlex.quote(data['nodes'][side].get('hf_command') or 'hf')
        private_file(path / (role + '.env'), render(values))
        rules = '\n'.join('sudo ' + shlex.join(rule) for rule in data['firewall'][side])
        private_file(path / (role + '.ufw.sh'), '#!/usr/bin/env bash\nset -euo pipefail\n'
                     '# Review before running. Adds peer rules; never enables, disables or resets UFW.\n' + rules + '\n')
        preparation = ('#!/usr/bin/env bash\nset -euo pipefail\n'
                       '# Run as the deployment user on this node; reuses its existing HF cache.\n'
                       'export HF_HUB_CACHE=' + shlex.quote(data['nodes'][side]['hub']) + '\n'
                       'docker pull ' + shlex.quote(values['IMAGE']) + '\n' +
                       cli + ' download ' + shlex.quote(values['MODEL_ID']) + ' --revision ' + values['MODEL_REVISION'] + '\n' +
                       cli + ' download ' + shlex.quote(values['DRAFT_ID']) + ' --revision ' + values['DRAFT_REVISION'] + '\n')
        private_file(path / (role + '.prepare.sh'), preparation)


def payload_for(data, side):
    node, values = data['nodes'][side], data['values'][side]
    return dict(root=node['root'], source_sha256=node['source_sha256'], env=render(values),
                directories=[values[key] for key in ('CACHE_HOST_DIR', 'LOG_HOST_DIR', 'SNAPSHOT_SEED_DIR')])


def apply_plan(data):
    left, right = payload_for(data, 0), payload_for(data, 1)
    setup_probe.writable_checkout(Path(left['root']), left['source_sha256'])
    request(data['worker'], 'guard', right)
    # Worker first. If the later head write fails, report the partial result without deleting operator data.
    created = request(data['worker'], 'apply', right)
    try:
        setup_probe.apply(left)
    except Exception as error:
        raise ValueError(f'Worker settings created at {created["created"]}; head write failed: {error}. No container was started') from error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', help='SSH hostname or user@host (passwordless SSH required)')
    p.add_argument('--worker-dir', help='Absolute path of the same checkout on the worker; defaults to the head path')
    p.add_argument('--head-ip', help='Override the head IPv4 discovered from the SSH connection')
    p.add_argument('--worker-ip', help='Override the worker IPv4 discovered from the SSH connection')
    p.add_argument('--fabric-subnets', help='Optional two space-separated selectors; otherwise discovers shared RDMA networks')
    p.add_argument('--profile', choices=('c4', 'c6'), default='c4')
    p.add_argument('--image', default=IMAGE)
    p.add_argument('--model-id', default=MODEL_ID)
    p.add_argument('--model-revision', default=MODEL_REVISION)
    p.add_argument('--draft-id', default=DRAFT_ID)
    p.add_argument('--draft-revision', default=DRAFT_REVISION)
    p.add_argument('--client-cidr', help='Optional trusted client IPv4/CIDR for head API port 8000')
    p.add_argument('--apply', action='store_true', help='Create fresh .env settings on both nodes; refuses existing settings/live mounts')
    p.add_argument('--prepare-assets', action='store_true', help='With --apply, pull the image and download/reuse the pinned models on both nodes')
    p.add_argument('--check-network', action='store_true', help='Test temporary TCP listeners in both directions on all three link pairs')
    a = p.parse_args()
    try:
        if a.prepare_assets and not a.apply:
            raise ValueError('--prepare-assets requires --apply for a fresh setup; use the generated *.prepare.sh to resume downloads')
        if not a.worker:
            if not sys.stdin.isatty():
                p.error('Supply --worker user@host (or run interactively)')
            a.worker = input('Worker SSH destination (for example user@worker): ').strip()
            if not a.worker_dir:
                a.worker_dir = input(f'Worker checkout [{ROOT}]: ').strip() or str(ROOT)
        validate_worker(a.worker)
        worker_dir = a.worker_dir or str(ROOT)
        if not Path(worker_dir).is_absolute() or '\n' in worker_dir or '\r' in worker_dir:
            raise ValueError('--worker-dir must be an absolute path without newlines')
        head = setup_probe.inventory(ROOT)
        worker = request(a.worker, 'inventory', {'root': worker_dir})
        data = plan(head, worker, a.worker, a.head_ip, a.worker_ip,
                    a.fabric_subnets.split() if a.fabric_subnets else None, a.profile, a.image, a.client_cidr,
                    {key: getattr(a, key) for key in ('model_id', 'model_revision', 'draft_id', 'draft_revision')})
        path = ROOT / 'site' / ('setup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(3))
        write_plan(data, path)
        print('Private setup plan:', path)
        print(f'Profile {a.profile.upper()}: 8 GiB KV, 1,047,552 context, API 8000; models remain in each node\'s HF cache.')
        for label, pair in zip(('management', 'RoCE 1', 'RoCE 2'), data['pairs']):
            print(f'  {label}: {pair[0]["interface"]} {pair[0]["address"]} <-> {pair[1]["interface"]} {pair[1]["address"]}')
        for warning in data['warnings']:
            print('ATTENTION:', warning)
        if not a.client_cidr:
            print('No client API rule generated. Use --client-cidr with your trusted client address/network if needed.')
        print('Review head.env / worker.env and *.ufw.sh. Firewall rules are never applied automatically.')
        if a.check_network:
            for result in network_checks(a.worker, data['pairs']):
                print(result)
            print('TCP reachability passed; startup still validates RoCE/GIDs and GPU collectives.')
        if a.apply:
            required = ('docker', 'compose', 'gpu_driver')
            for node in data['nodes']:
                if not all(node['checks'][key]['ok'] for key in required) or not all(
                        node['checks'][key] for key in ('arm64', 'headless')) or node['checks']['drm_modeset'] is False:
                    raise ValueError('Host prerequisites incomplete; see plan warnings and docs/DEPLOYMENT.md')
                if a.prepare_assets and not node['checks']['tools'].get('hf'):
                    raise ValueError('Asset preparation needs the Hugging Face CLI on both nodes. Install it, or omit --prepare-assets if the image/models are already cached. No settings were written')
            apply_plan(data)
            print('Created both .env files (mode 600). No container was started.', flush=True)
            if a.prepare_assets:
                print('Preparing the worker image/models in its existing cache...', flush=True)
                subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', a.worker, 'bash -s'],
                               input=(path / 'worker.prepare.sh').read_text(), text=True, check=True)
                print('Preparing the head image/models in its existing cache...', flush=True)
                subprocess.run(['bash', str(path / 'head.prepare.sh')], check=True)
            else:
                print('Prepare assets: bash ' + shlex.quote(str(path / 'head.prepare.sh')))
                print('Worker assets: ' + shlex.join(['ssh', a.worker, 'bash -s']) + ' < ' + shlex.quote(str(path / 'worker.prepare.sh')))
            print('When both downloads finish, run ./check.sh and ./start.sh --approved.')
        else:
            print('Plan only: no deployment settings, firewall, model files or containers were changed.')
            print('Re-run with --apply to create fresh settings; add --check-network after reviewing firewall access.')
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print('Setup failed:', error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
