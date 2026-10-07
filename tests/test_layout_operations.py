"""Exercise root commands and sync guards with fake Docker/SSH; no GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sync_source', ROOT / 'scripts/sync-repo.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class LayoutOperations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'deployment with spaces'
        self.root.mkdir()
        (self.root / 'scripts').mkdir()
        shutil.copy2(ROOT / 'scripts/cluster.sh', self.root / 'scripts/cluster.sh')
        for name in ['start.sh', 'stop.sh', 'restart.sh', 'status.sh', 'tail-log.sh', 'check.sh', 'sync-repo.sh', 'verify.sh']:
            shutil.copy2(ROOT / name, self.root / name)
        self.trace = Path(self.temp.name) / 'trace'
        self.bin = Path(self.temp.name) / 'bin'
        self.bin.mkdir()
        for command in ['docker', 'ssh', 'python3', 'curl', 'flock']:
            path = self.bin / command
            path.write_text('#!/bin/bash\nprintf "%s\\0" "${0##*/}" "$@" >> "$TRACE"\nprintf "\\n" >> "$TRACE"\n')
            path.chmod(0o755)
        (self.root / '.env').write_text("ROLE=head\nPEER_SSH=user@worker\nPEER_DEPLOY_DIR='/path with spaces/worker'\nHEAD_HOST=10.0.0.1\nLOG_HOST_DIR=" + shlex.quote(str(self.root)) + '\n')
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'], TRACE=str(self.trace))

    def run_root(self, name, *args):
        return subprocess.run(['bash', str(self.root / name), *args], cwd='/', env=self.env, capture_output=True, text=True)

    def calls(self):
        return [line.decode().rstrip('\0').split('\0') for line in self.trace.read_bytes().splitlines()] if self.trace.exists() else []

    def test_start_without_approval_never_calls_docker(self):
        result = self.run_root('start.sh')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_start_uses_shared_core_and_preserves_native_sequence(self):
        result = self.run_root('start.sh', '--approved')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertTrue(any(c[:2] == ['python3', str(self.root / 'scripts/check.py')] for c in calls))
        docker = next(c for c in calls if c[0] == 'docker')
        self.assertIn(str(self.root / 'compose.yaml'), docker)
        self.assertEqual(docker[-2:], ['up', '-d'])
        remote = [shlex.split(c[-1]) for c in calls if c[0] == 'ssh']
        self.assertIn(['bash', '/path with spaces/worker/scripts/cluster.sh', 'node-up', '--approved'], remote)

    def test_worker_logs_preserve_ssh_path_and_arguments(self):
        result = self.run_root('tail-log.sh', 'worker', '--tail', '20')
        self.assertEqual(result.returncode, 0, result.stderr)
        call = next(c for c in self.calls() if c[0] == 'ssh')
        self.assertEqual(shlex.split(call[-1]), ['bash', '/path with spaces/worker/scripts/cluster.sh', 'node-logs', '--tail', '20'])

    def test_stop_stops_both_ranks(self):
        result = self.run_root('stop.sh', '--approved')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertTrue(any(c[0] == 'docker' and c[-3:] == ['down', '--timeout', '60'] for c in calls))
        remote = [shlex.split(c[-1]) for c in calls if c[0] == 'ssh']
        self.assertIn(['bash', '/path with spaces/worker/scripts/cluster.sh', 'node-down', '--approved'], remote)

    def test_verify_uses_relocated_test_assets(self):
        result = self.run_root('verify.sh')
        # bash invokes the fixture-missing smoketest after the first relocated check.
        self.assertTrue(any(c[:2] == ['python3', str(self.root / 'tests/verify-api.py')] for c in self.calls()))
        self.assertIn('tests/smoketest/run.sh', result.stderr)

    def test_sync_selection_preserves_private_config_and_metadata(self):
        names = ['.env', '.env.worker', '.env-BAK', '.env.example', '.git/config', 'logs/x',
                 'experiments/report.json', 'backups/private.tar', 'models/weights', '../escape',
                 'README.md', 'files/entrypoint.sh', '.gitignore']
        self.assertEqual(sync.tracked_source(names), ['.env.example', 'README.md', 'files/entrypoint.sh', '.gitignore'])

    def test_manifest_refresh_ignores_private_untracked_files(self):
        shutil.copy2(ROOT / 'scripts/update-manifest.py', self.root / 'scripts/update-manifest.py')
        (self.root / 'manifests').mkdir()
        (self.root / 'manifests/source.json').write_text('{}')
        (self.root / 'files').mkdir()
        (self.root / 'files/entrypoint.sh').write_text('# runtime')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), 'add', 'files/entrypoint.sh', 'scripts/update-manifest.py'], check=True)
        (self.root / 'backups').mkdir()
        (self.root / 'backups/private.tar.gz').write_bytes(b'private backup')
        (self.root / '.env-BAK').write_text('private config')
        result = subprocess.run([sys.executable, str(self.root / 'scripts/update-manifest.py')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.root / 'manifests/source.json').read_text())
        self.assertEqual(set(manifest['files']), {'files/entrypoint.sh', 'scripts/update-manifest.py'})
        self.assertEqual(manifest['native_entrypoint_sha256'], manifest['files']['files/entrypoint.sh'])

    def test_sync_guard_rejects_live_mounted_source(self):
        docker = self.bin / 'docker'
        containers = [{'Name': '/live', 'Mounts': [{'Source': str(self.root / 'files/entrypoint.sh')}]}]
        docker.write_text('#!/bin/bash\nif [[ "$1" == ps ]]; then echo abc; else cat "$FIXTURE"; fi\n')
        fixture = Path(self.temp.name) / 'containers.json'
        fixture.write_text(json.dumps(containers))
        env = dict(self.env, FIXTURE=str(fixture))
        result = subprocess.run([sys.executable, '-c', sync.GUARD, str(self.root)], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Refusing sync', result.stderr)
        fixture.write_text('[]')
        result = subprocess.run([sys.executable, '-c', sync.GUARD, str(self.root)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(json.loads(result.stdout)['env_sha256'])


if __name__ == '__main__':
    unittest.main()
