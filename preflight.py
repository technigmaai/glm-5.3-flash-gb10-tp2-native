#!/usr/bin/env python3
"""Discover configured RoCE links and check idle resources before native startup."""
import ipaddress
import json
from pathlib import Path
import socket
import subprocess
import sys
from settings import load_settings


def fabric_addresses(addresses, prefixes):
    return {(link['ifname'], item['local']) for link in addresses
            for item in link.get('addr_info', [])
            if item['family'] == 'inet' and any(item['local'].startswith(p) for p in prefixes)}


def main():
    settings = load_settings()
    errors = []
    addresses = json.loads(subprocess.check_output(['ip', '-j', '-4', 'addr'], text=True))
    links = fabric_addresses(addresses, settings['FABRIC_SUBNETS'].split())
    if len({name for name, _ in links}) < int(settings.get('FABRIC_MIN_DEVICES', '2')):
        errors.append('Not all configured fabric links are present')
    gids = []
    for iface, address in sorted(links):
        root = Path('/sys/class/net') / iface
        if (root / 'operstate').read_text().strip() != 'up':
            errors.append(f'{iface}: not UP')
        if int((root / 'mtu').read_text()) < int(settings.get('FABRIC_MTU', '9000')):
            errors.append(f'{iface}: MTU below the configured minimum')
        found = []
        for path in Path('/sys/class/infiniband').glob('*/ports/1/gid_attrs/ndevs/*'):
            try:
                if path.read_text().strip() != iface:
                    continue
                port = path.parents[2]; idx = path.name
                typ = (port / 'gid_attrs/types' / idx).read_text().strip()
                gid = ipaddress.IPv6Address((port / 'gids' / idx).read_text().strip())
                if typ == 'RoCE v2' and gid.ipv4_mapped == ipaddress.IPv4Address(address):
                    found.append(idx)
            except (OSError, ValueError):
                continue
        if len(found) != 1:
            errors.append(f'{iface}: expected one RoCE v2 GID for {address}, got {found}')
        else:
            gids.append(found[0])
    if len(set(gids)) > 1:
        errors.append(f'Fabric roots have different GID indexes: {gids}')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    if gpu:
        errors.append('A GPU workload is active: ' + gpu)
    ports = [int(settings.get(k, default)) for k, default in
             (('STATUS_PORT', '8082'), ('MASTER_PORT', '29553'), ('FABRIC_CHECK_PORT', '29511'))]
    if settings['ROLE'] == 'head':
        ports.append(int(settings.get('API_PORT', '8000')))
    for port in ports:
        with socket.socket() as sock:
            # Match server bind semantics: closed TCP sockets may be in TIME_WAIT.
            # SO_REUSEADDR still rejects an active listener on this address/port.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('0.0.0.0', port))
            except OSError:
                errors.append(f'Port {port} is occupied')
    memory = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if len(line.split()) >= 3 and line.split()[1].isdigit()}
    if memory['MemAvailable'] < float(settings.get('MIN_AVAILABLE_GIB', '100')) * 2**30:
        errors.append(f'Only {memory["MemAvailable"]/2**30:.1f} GiB available')
    if subprocess.run(['systemctl', 'is-active', '--quiet', 'display-manager']).returncode == 0:
        errors.append('The display manager is active; this profile requires headless nodes')
    if errors:
        print('\n'.join(errors), file=sys.stderr); return 1
    print(f'Preflight PASS: {len(links)} fabric links, GID {gids[0]}, idle GPU and free ports')
    return 0


if __name__ == '__main__':
    sys.exit(main())
