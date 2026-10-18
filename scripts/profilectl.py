#!/usr/bin/env python3
"""Select a complete, versioned native TP2 recipe on both nodes."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from model_source import compose_command
from runtime_profiles import catalog, image_ids, selected
from settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
NODE_LOCAL_ENV = {'ROLE', 'NODE_RANK', 'VLLM_HOST_IP', 'FABRIC_SUBNETS',
                  'NCCL_IB_HCA', 'NCCL_IB_GID_INDEX', 'NCCL_SOCKET_IFNAME'}


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs).stdout


@contextmanager
def version_environment(version):
    # Keep the host setting helpers consistent without changing any .env file.
    before = os.environ.get('GLM53_PROFILE')
    try:
        if version:
            os.environ['GLM53_PROFILE'] = version
        else:
            os.environ.pop('GLM53_PROFILE', None)
        yield
    finally:
        if before is None:
            os.environ.pop('GLM53_PROFILE', None)
        else:
            os.environ['GLM53_PROFILE'] = before


class Node:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.versions = catalog(self.root)

    def settings(self, version=None):
        with version_environment(version):
            values = load_settings(self.root)
        if Path(values['DEPLOY_ROOT']).resolve() != self.root:
            raise ValueError('DEPLOY_ROOT must point to this checkout: ' + str(self.root))
        if version:
            base = self.settings()
            for key in ('ROLE', 'NODE_RANK', 'HEAD_HOST', 'PEER_SSH', 'PEER_DEPLOY_DIR',
                        'PROJECT_NAME', 'CONTAINER_NAME', 'CLUSTER_LOCK', 'LOG_HOST_DIR'):
                if values.get(key) != base.get(key):
                    raise ValueError('Version overrides cannot change node/cluster identity: ' + key)
        return values

    def compose(self, version):
        with version_environment(version):
            s = self.settings(version)
            base = self.settings()
            for key, default in (('PROJECT_NAME', 'glm53-native'), ('CONTAINER_NAME', 'glm53-native')):
                if (s.get(key) or default) != (base.get(key) or default):
                    raise ValueError('Version overrides cannot change ' + key)
            return compose_command(self.root, s)

    def inspect(self):
        s = self.settings()
        name = s.get('CONTAINER_NAME') or 'glm53-native'
        p = subprocess.run(['docker', 'container', 'inspect', name], text=True, capture_output=True)
        if p.returncode:
            if ('no such container:' in p.stderr.lower() or 'no such object:' in p.stderr.lower()):
                return None
            raise ValueError(p.stderr.strip() or 'Cannot inspect ' + name)
        d = json.loads(p.stdout)[0]
        labels = d['Config'].get('Labels') or {}
        version = next((v for v, spec in self.versions.items() if d['Image'] in image_ids(spec)), None)
        owned = (labels.get('com.docker.compose.project') == (s.get('PROJECT_NAME') or 'glm53-native')
                 and labels.get('com.docker.compose.service') == 'glm53')
        if labels.get('io.technigmaai.profile') not in (None, version):
            owned = False
        return {'version': version, 'recognized': owned and version is not None,
                'image': d['Config']['Image'], 'image_id': d['Image'],
                'status': d['State']['Status'], 'health': d['State'].get('Health', {}).get('Status'),
                'started_at': d['State']['StartedAt'], 'restart_count': d['RestartCount'],
                'recipe': labels.get('io.technigmaai.profile'),
                'working_dir': labels.get('com.docker.compose.project.working_dir')}

    def prepare(self, version):
        spec = self.versions[version]
        s = self.settings(version)
        role = s['ROLE']
        if role not in ('head', 'worker') or s.get('NODE_RANK') != ('0' if role == 'head' else '1'):
            raise ValueError('ROLE and NODE_RANK must identify head/0 or worker/1')
        image = json.loads(run(['docker', 'image', 'inspect', spec['image']]))[0]
        if image['Id'] not in image_ids(spec):
            raise ValueError('Image identity differs for ' + version + '; no containers changed')
        if image.get('Architecture') != 'arm64':
            raise ValueError('This recipe requires the ARM64 GB10 runtime')
        with version_environment(version):
            run([sys.executable, str(self.root / 'scripts/check.py')], env=dict(os.environ))
            service = json.loads(run(self.compose(version) + ['config', '--format', 'json']))['services']['glm53']
        return {'version': version, 'role': role, 'image_id': image['Id'],
                'config_digest': spec['config_digest'],
                'source_sha256': hashlib.sha256((self.root / 'manifests/source.json').read_bytes()).hexdigest(),
                'project': s.get('PROJECT_NAME') or 'glm53-native',
                'container': s.get('CONTAINER_NAME') or 'glm53-native',
                'peer_dir': s.get('PEER_DEPLOY_DIR'),
                'shared': {k: v for k, v in service['environment'].items() if k not in NODE_LOCAL_ENV}}

    def action(self, action, version):
        if action == 'snapshot':
            return self.inspect()
        if action == 'prepare':
            return self.prepare(version)
        if action == 'preflight':
            with version_environment(version):
                run([sys.executable, str(self.root / 'scripts/preflight.py')], env=dict(os.environ))
            return {'preflight': 'passed'}
        if action == 'config':
            return json.loads(run(self.compose(version) + ['config', '--format', 'json']))
        # Internal mutation entry point is gated independently of the head CLI.
        raise ValueError('Unsupported node action: ' + action)


class Controller:
    def __init__(self, root=ROOT, invoke=None):
        self.node = Node(root)
        self.settings = self.node.settings()
        if self.settings['ROLE'] != 'head':
            raise ValueError('Run cluster launchers on the head node')
        self.peer = self.settings['PEER_SSH']
        self.peer_dir = self.settings['PEER_DEPLOY_DIR']
        if not self.peer or self.peer.startswith('-') or any(c.isspace() for c in self.peer):
            raise ValueError('PEER_SSH must be a valid SSH destination')
        if not Path(self.peer_dir).is_absolute():
            raise ValueError('PEER_DEPLOY_DIR must be absolute')
        self.invoke = invoke or self.call

    def call(self, role, action, version=None):
        if role == 'head':
            if action in ('up', 'down'):
                return mutate(self.node, action, version)
            return self.node.action(action, version)
        command = ['python3', str(Path(self.peer_dir) / 'scripts/profilectl.py'), '_node', action]
        if version:
            command += ['--profile', version]
        if action in ('up', 'down'):
            command += ['--approved']
        output = run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', self.peer, shlex.join(command)])
        return json.loads(output)

    def snapshots(self):
        return {role: self.invoke(role, 'snapshot') for role in ('head', 'worker')}

    @staticmethod
    def active(states):
        if any(d and not d['recognized'] for d in states.values()):
            raise ValueError('Unrecognized deployment; no containers changed')
        versions = {d['version'] for d in states.values() if d}
        if len(versions) > 1:
            raise ValueError('Nodes have different runtime versions; no containers changed')
        return next(iter(versions), None)

    def prepare(self, version):
        nodes = {r: self.invoke(r, 'prepare', version) for r in ('head', 'worker')}
        if nodes['head']['role'] != 'head' or nodes['worker']['role'] != 'worker':
            raise ValueError('Peer roles differ; no containers changed')
        for k in ('version', 'config_digest', 'source_sha256', 'project', 'container', 'shared'):
            if nodes['head'][k] != nodes['worker'][k]:
                raise ValueError('Head/worker configuration differs: ' + k + '; no containers changed')
        # All variants use the base identity, ports and lifecycle lock.
        for d in nodes.values():
            if (d['project'] != (self.settings.get('PROJECT_NAME') or 'glm53-native')
                    or d['container'] != (self.settings.get('CONTAINER_NAME') or 'glm53-native')):
                raise ValueError('Version overrides cannot change the project or container identity')
        if nodes['head']['peer_dir'] != self.peer_dir:
            raise ValueError('Version overrides cannot change PEER_DEPLOY_DIR')
        return nodes

    @contextmanager
    def locked(self):
        import fcntl
        lock = Path(self.settings.get('CLUSTER_LOCK') or (self.settings['LOG_HOST_DIR'] + '/cluster.lock'))
        with lock.open('a') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another native cluster operation is in progress')
            yield

    def change(self, action, version=None, dry_run=False):
        with self.locked():
            states = self.snapshots()
            active = self.active(states)
            target = version or active or self.settings.get('RUNTIME_VERSION', 'v0.1.0')
            selected(self.node.root, target)
            if action != 'stop':
                self.prepare(target)  # Validate BOTH nodes before stopping either rank.
            roots = {'head': str(self.node.root), 'worker': self.peer_dir}
            if action == 'start' and all(d and d['status'] == 'running' and d['version'] == target
                                        and d['recipe'] == target and d['working_dir'] == roots[role]
                                        for role, d in states.items()):
                print('Already running ' + target + '. Use restart to apply source/configuration changes.')
                return
            if action == 'stop' and version and active not in (None, version):
                raise ValueError('A different version is selected; nothing stopped')
            steps = []
            if active:
                steps += [(r, 'down', target) for r in ('head', 'worker') if states[r]]
            if action != 'stop':
                steps += [(r, 'preflight', target) for r in ('head', 'worker')]
                steps += [(r, 'up', target) for r in ('head', 'worker')]
            if dry_run:
                print(json.dumps({'dry_run': True, 'version': target, 'steps': steps}, indent=2))
                return
            for role, step, v in steps:
                print(f'{role}: {step} {v}', flush=True)
                self.invoke(role, step, v)
            print('Stopped both ranks.' if action == 'stop' else 'Launching ' + target + '. Run ./status.sh to check readiness.')


def mutate(node, action, version):
    # Recheck ownership immediately before each container operation.
    current = node.inspect()
    if current and not current['recognized']:
        raise ValueError('Refusing to change an unrecognized container')
    if action == 'up':
        node.prepare(version)
    args = ['up', '-d', '--pull', 'never', '--no-build'] if action == 'up' else ['down', '--timeout', '60']
    run(node.compose(version) + args)
    return {'action': action, 'version': version}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('list', 'start', 'restart', 'stop', 'status', 'check', 'config', '_node'))
    parser.add_argument('node_action', nargs='?', choices=('snapshot', 'prepare', 'preflight', 'config', 'up', 'down'))
    parser.add_argument('--profile', choices=list(catalog(ROOT)))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--approved', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--all', action='store_true', help='check every published version')
    args = parser.parse_args()
    if args.all and (args.action != 'check' or args.profile):
        parser.error('--all is only for check and cannot be combined with --profile')
    if args.action == 'list':
        if args.json:
            print(json.dumps(catalog(ROOT), indent=2))
        else:
            for v, spec in catalog(ROOT).items():
                print(v + '  ' + spec['description'] + '\n  ' + spec['image'])
        return
    if (args.action in ('start', 'restart', 'stop') or args.action == '_node' and args.node_action in ('up', 'down')) and not (args.approved or args.dry_run):
        parser.error('This changes containers. Supply --approved, or inspect --dry-run first.')
    if args.action == '_node':
        if not args.node_action:
            parser.error('_node requires an action')
        if args.node_action not in ('snapshot',) and not args.profile:
            parser.error('node action requires --profile')
        node = Node()
        if args.node_action in ('up', 'down'):
            if not args.approved:
                parser.error('Internal mutations require --approved')
            result = mutate(node, args.node_action, args.profile)
        else:
            result = node.action(args.node_action, args.profile)
        print(json.dumps(result))
        return
    c = Controller()
    if args.action in ('start', 'restart', 'stop'):
        version = args.profile
        if args.action == 'start' and not version:
            version = c.settings.get('RUNTIME_VERSION', 'v0.1.0')
        c.change(args.action, version, args.dry_run)
    elif args.action == 'status':
        states = c.snapshots()
        if args.json:
            print(json.dumps(states, indent=2))
        else:
            for role, d in states.items():
                print(role + ': absent' if not d else role + ': ' + (d['version'] or 'unrecognized') + ' | ' + d['status'] + ' | ' + str(d['health']))
            print('API: http://' + c.settings['HEAD_HOST'] + ':' + c.settings.get('API_PORT', '8000') + '/v1')
    elif args.action == 'check':
        versions = list(c.node.versions) if args.all else [args.profile or c.settings.get('RUNTIME_VERSION', 'v0.1.0')]
        for v in versions:
            c.prepare(v)
            print(v + ': both-node image, source, configuration and model checks passed')
    elif args.action == 'config':
        v = args.profile or c.settings.get('RUNTIME_VERSION', 'v0.1.0')
        print(json.dumps(c.node.action('config', v), indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        sys.exit(detail)
