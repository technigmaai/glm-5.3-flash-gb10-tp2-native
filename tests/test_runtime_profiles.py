"""Exercise two-node switching and every merged recipe without starting a GPU."""
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import profilectl
from runtime_profiles import catalog, env_file
from settings import load_settings


def snapshot(version='v0.4.0', recognized=True, recipe=None):
    return {'version': version, 'recognized': recognized, 'status': 'running', 'recipe': recipe}


class Switching(unittest.TestCase):
    def setUp(self):
        output = patch('sys.stdout', new_callable=io.StringIO)
        output.start()
        self.addCleanup(output.stop)

    def controller(self, states=None, failure=None, mismatch=None):
        self.calls = []
        states = states or {'head': snapshot(), 'worker': snapshot()}
        for role, state in states.items():
            if state is not None:
                state.setdefault('working_dir', str(ROOT) if role == 'head' else '/worker checkout')
        c = profilectl.Controller.__new__(profilectl.Controller)
        c.node = SimpleNamespace(root=ROOT, versions=catalog(ROOT))
        c.settings = {'PROJECT_NAME': 'glm53-native', 'CONTAINER_NAME': 'glm53-native'}
        c.peer_dir = '/worker checkout'
        c.locked = nullcontext
        def invoke(role, action, version=None):
            self.calls.append((role, action, version))
            if failure == (role, action):
                raise ValueError('missing worker image')
            if action == 'snapshot':
                return states[role]
            if action == 'prepare':
                return {'role': role, 'version': version, 'image_id': 'sha256:' + role,
                        'config_digest': 'sha256:canonical-config',
                        'source_sha256': ('different' if role == 'worker' and mismatch else 'same'),
                        'project': 'glm53-native', 'container': 'glm53-native',
                        'peer_dir': c.peer_dir, 'shared': {'TP': '2'}}
            return {}
        c.invoke = invoke
        return c

    def test_switch_validates_both_then_stops_both_then_preflights_both(self):
        c = self.controller()
        c.change('start', 'v0.6.0')
        self.assertEqual([(r, a) for r, a, _ in self.calls], [
            ('head', 'snapshot'), ('worker', 'snapshot'), ('head', 'prepare'), ('worker', 'prepare'),
            ('head', 'down'), ('worker', 'down'), ('head', 'preflight'), ('worker', 'preflight'),
            ('head', 'up'), ('worker', 'up')])

    def test_missing_worker_image_cannot_stop_the_head(self):
        c = self.controller(failure=('worker', 'prepare'))
        with self.assertRaisesRegex(ValueError, 'missing worker image'):
            c.change('start', 'v0.6.0')
        self.assertFalse(any(a in ('down', 'up') for _, a, _ in self.calls))

    def test_different_source_revisions_cannot_stop_either_node(self):
        c = self.controller(mismatch=True)
        with self.assertRaisesRegex(ValueError, 'source_sha256'):
            c.change('start', 'v0.6.0')
        self.assertFalse(any(a == 'down' for _, a, _ in self.calls))

    def test_unknown_and_mixed_pairs_refuse_changes(self):
        for states in [dict(head=snapshot(recognized=False), worker=snapshot()),
                       dict(head=snapshot('v0.3.0'), worker=snapshot('v0.4.0'))]:
            with self.subTest(states=states):
                c = self.controller(states)
                with self.assertRaises(ValueError):
                    c.change('start', 'v0.6.0')
                self.assertEqual(len(self.calls), 2)

    def test_dry_run_has_no_container_mutations_or_idle_checks(self):
        c = self.controller()
        c.change('start', 'v0.6.0', dry_run=True)
        self.assertEqual([a for _, a, _ in self.calls], ['snapshot', 'snapshot', 'prepare', 'prepare'])

    def test_repeated_version_leaves_running_pair_untouched(self):
        c = self.controller(dict(head=snapshot('v0.6.0', recipe='v0.6.0'),
                                 worker=snapshot('v0.6.0', recipe='v0.6.0')))
        c.change('start', 'v0.6.0')
        self.assertFalse(any(a in ('up', 'down') for _, a, _ in self.calls))

    def test_existing_unlabelled_native_pair_gets_the_selected_recipe(self):
        c = self.controller(dict(head=snapshot('v0.6.0'), worker=snapshot('v0.6.0')))
        c.change('start', 'v0.6.0')
        self.assertEqual(sum(a == 'up' for _, a, _ in self.calls), 2)

    def test_same_version_from_another_checkout_is_not_mistaken_for_this_recipe(self):
        states = dict(head=snapshot('v0.6.0', recipe='v0.6.0'), worker=snapshot('v0.6.0', recipe='v0.6.0'))
        states['head']['working_dir'] = '/previous checkout'
        c = self.controller(states)
        c.change('start', 'v0.6.0')
        self.assertEqual(sum(a == 'up' for _, a, _ in self.calls), 2)

    def test_preflight_failure_never_starts_either_rank(self):
        c = self.controller(failure=('worker', 'preflight'))
        with self.assertRaises(ValueError):
            c.change('start', 'v0.6.0')
        self.assertFalse(any(a == 'up' for _, a, _ in self.calls))

    def test_stop_uses_running_version_and_refuses_another_version(self):
        c = self.controller()
        with self.assertRaisesRegex(ValueError, 'different version'):
            c.change('stop', 'v0.6.0')
        c = self.controller()
        c.change('stop')
        self.assertEqual(self.calls[-2:], [('head', 'down', 'v0.4.0'), ('worker', 'down', 'v0.4.0')])


class Recipes(unittest.TestCase):
    def test_container_identity_works_with_classic_and_containerd_image_stores(self):
        node = profilectl.Node.__new__(profilectl.Node)
        node.versions = catalog(ROOT)
        node.settings = lambda: {'PROJECT_NAME': 'glm53-native'}
        spec = node.versions['v0.6.0']
        for key in ('index_digest', 'platform_digest', 'config_digest'):
            data = {'Image': spec[key], 'Config': {'Image': spec['image'], 'Labels': {
                'com.docker.compose.project': 'glm53-native', 'com.docker.compose.service': 'glm53'}},
                'State': {'Status': 'running', 'StartedAt': 'unchanged'}, 'RestartCount': 0}
            with patch.object(profilectl.subprocess, 'run', return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps([data]), stderr='')):
                self.assertEqual(node.inspect()['version'], 'v0.6.0')

    def test_a_version_tag_on_different_image_contents_is_not_trusted(self):
        node = profilectl.Node.__new__(profilectl.Node)
        node.versions = catalog(ROOT)
        node.settings = lambda: {'PROJECT_NAME': 'glm53-native'}
        data = {'Image': 'sha256:unexpected', 'Config': {'Image': node.versions['v0.6.0']['image'], 'Labels': {
            'com.docker.compose.project': 'glm53-native', 'com.docker.compose.service': 'glm53'}},
            'State': {'Status': 'running', 'StartedAt': 'unchanged'}, 'RestartCount': 0}
        with patch.object(profilectl.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps([data]), stderr='')):
            self.assertFalse(node.inspect()['recognized'])

    def test_six_consecutive_versions_have_distinct_image_identities(self):
        versions = catalog(ROOT)
        self.assertEqual(list(versions), [f'v0.{n}.0' for n in range(1, 7)])
        self.assertEqual(len({s['config_digest'] for s in versions.values()}), 6)
        for v, spec in versions.items():
            self.assertTrue(spec['image'].endswith(':' + v))
        self.assertEqual([p.name for p in (ROOT / 'launchers').glob('*.sh')], ['launch.sh'])

    def test_recipe_mounts_are_self_contained_unique_and_read_only(self):
        manifest = json.loads((ROOT / 'manifests/source.json').read_text())['files']
        for v, spec in catalog(ROOT).items():
            service = json.loads((ROOT / spec['compose']).read_text())['services']['glm53']
            targets = []
            for mount in service.get('volumes', []):
                path = mount['source'].split('}/', 1)[1]
                self.assertTrue((ROOT / path).is_file(), path)
                self.assertIn(path, manifest, 'A fresh clone must contain every runtime patch')
                self.assertTrue(mount['read_only'])
                self.assertFalse(mount['bind']['create_host_path'])
                targets.append(mount['target'])
            self.assertEqual(len(targets), len(set(targets)), v)
            self.assertEqual(service['image'], spec['image'])

    def test_imported_payloads_match_pinned_provenance(self):
        for layer in ('sixpatch', 'trio', 'sampling'):
            m = json.loads((ROOT / 'profiles/provenance' / (layer + '.json')).read_text())
            for entry in m['files']:
                f = ROOT / 'profiles/patches' / layer / entry['path']
                self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(), entry['payload_sha256'])

    def test_explicit_version_does_not_rewrite_node_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'profiles').mkdir()
            shutil.copy2(ROOT / 'profiles/images.json', root / 'profiles/images.json')
            original = 'ROLE=head\nIMAGE=keep-original\nMAX_NUM_SEQS=6\n'
            (root / '.env').write_text(original)
            with patch.dict(os.environ, GLM53_PROFILE='v0.6.0'):
                self.assertEqual(load_settings(root)['IMAGE'], catalog(ROOT)['v0.6.0']['image'])
                self.assertEqual(load_settings(root)['MAX_NUM_SEQS'], '6')
                self.assertEqual(env_file(root), root / '.env')
            (root / '.env.v0.6.0').write_text('ROLE=head\nMAX_NUM_SEQS=16\n')
            with patch.dict(os.environ, GLM53_PROFILE='v0.6.0'):
                self.assertEqual(load_settings(root)['MAX_NUM_SEQS'], '16')
            self.assertEqual((root / '.env').read_text(), original)

    def test_mutation_without_approval_fails_before_reading_private_settings(self):
        for action in ('start', 'stop', 'restart'):
            p = subprocess.run([sys.executable, str(ROOT / 'scripts/profilectl.py'), action],
                               text=True, capture_output=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn('Supply --approved', p.stderr)
        p = subprocess.run([sys.executable, str(ROOT / 'scripts/profilectl.py'), '_node',
                            'up', '--profile', 'v0.6.0', '--dry-run'], text=True, capture_output=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn('Internal mutations require --approved', p.stderr)

    @unittest.skipUnless(shutil.which('docker'), 'Docker Compose CLI is required to render recipes')
    def test_all_six_profiles_merge_with_actual_compose(self):
        settings = {'ROLE': 'head', 'NODE_RANK': '0', 'HEAD_HOST': '10.0.0.1',
                    'VLLM_HOST_IP': '10.0.0.1', 'FABRIC_SUBNETS': '10.1.0. 10.2.0.',
                    'DEPLOY_ROOT': str(ROOT), 'IMAGE': 'unused:baseline',
                    'CACHE_HOST_DIR': '/tmp/glm-cache', 'LOG_HOST_DIR': '/tmp/glm-logs',
                    'HF_HUB_HOST_DIR': '/tmp/hf-hub', 'SNAPSHOT_SEED_DIR': '/tmp/seeds',
                    'MODEL_ID': 'nvidia/GLM-5.3-Flash-NVFP4', 'LIMIT_MM': '{"image":32,"video":0}'}
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write('\n'.join(k + '=' + shlex.quote(v) for k, v in settings.items())); f.flush()
            for v, spec in catalog(ROOT).items():
                command = ['docker', 'compose', '--env-file', f.name, '-f', str(ROOT / 'compose.yaml'),
                           '-f', str(ROOT / 'compose.models-hf.yaml'), '-f', str(ROOT / spec['compose']),
                           'config', '--format', 'json']
                service = json.loads(subprocess.check_output(command, text=True))['services']['glm53']
                self.assertEqual(service['image'], spec['image'])
                self.assertEqual(service['labels']['io.technigmaai.profile'], v)
                targets = {m['target']: m['source'] for m in service['volumes']}
                if v == 'v0.6.0':
                    self.assertIn('/profiles/patches/trio/', targets['/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/arx_vllm.cu'])
                    self.assertIn('/profiles/patches/sampling/', targets['/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py'])


if __name__ == '__main__':
    unittest.main()
