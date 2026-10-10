"""Fresh setup safety, adjacent fabric networks, cache mapping and real TCP probes."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import configure
import setup_probe
spec = importlib.util.spec_from_file_location('guided_setup', ROOT / 'scripts/setup.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def node(side):
    addresses = [('lan0', f'10.10.0.{side + 1}', 24, False),
                 ('roce0', f'192.168.2.{side + 1}', 30, True),
                 ('roce1', f'192.168.2.{side + 5}', 30, True)]
    return dict(root=f'/home/user{side}/deployment', home=f'/home/user{side}',
                hub=f'/cache/user{side}/hub', drm_gid=str(100 + side), source_sha256='abc',
                existing_env=False, ssh_connection='10.10.0.1 40000 10.10.0.2 22', ufw='active',
                addresses=[dict(ifname=iface, rdma=rdma, operstate='UP', mtu=9000,
                                addr_info=[dict(family='inet', local=address, prefixlen=prefix)])
                           for iface, address, prefix, rdma in addresses],
                checks=dict(docker=dict(ok=True, detail=''), compose=dict(ok=True, detail=''),
                            gpu_driver=dict(ok=True, detail=''), arm64=True, headless=True,
                            drm_modeset=True, tools=dict(hf=True, jq=True)))


class GuidedSetup(unittest.TestCase):
    def setUp(self):
        self.head, self.worker = node(0), node(1)

    def test_adjacent_30_networks_are_discovered_without_prefix_edits(self):
        data = setup.plan(self.head, self.worker, 'user@worker')
        self.assertEqual([pair[0]['network'] for pair in data['pairs'][1:]], ['192.168.2.0/30', '192.168.2.4/30'])
        self.assertEqual(data['values'][0]['FABRIC_SUBNETS'], '192.168.2.1 192.168.2.5')
        self.assertEqual(data['values'][1]['FABRIC_SUBNETS'], '192.168.2.2 192.168.2.6')

    def test_explicit_30_and_31_selectors(self):
        pairs = setup.fabric_pairs(self.head, self.worker, ['192.168.2.0/30', '192.168.2.4/30'])
        self.assertEqual(len(pairs), 2)
        for side, nd in enumerate((self.head, self.worker)):
            for index in (1, 2):
                nd['addresses'][index]['addr_info'][0].update(local=f'192.168.2.{(index - 1) * 2 + side}', prefixlen=31)
        self.assertEqual(len(setup.fabric_pairs(self.head, self.worker)), 2)

    def test_ambiguous_or_missing_links_fail_with_actionable_message(self):
        for side, nd in enumerate((self.head, self.worker)):
            extra = copy.deepcopy(nd['addresses'][1])
            extra['ifname'] = 'roce2'
            extra['addr_info'][0].update(local=f'10.22.0.{side + 1}', prefixlen=24)
            nd['addresses'].append(extra)
        with self.assertRaisesRegex(ValueError, '--fabric-subnets'):
            setup.fabric_pairs(self.head, self.worker)
        self.head, self.worker = node(0), node(1)
        self.worker['addresses'].pop()
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            setup.fabric_pairs(self.head, self.worker)

    def test_explicit_selection_does_not_allow_non_rdma_or_small_mtu(self):
        self.head['addresses'][1]['mtu'] = 1500
        with self.assertRaisesRegex(ValueError, 'MTU'):
            setup.fabric_pairs(self.head, self.worker)
        self.head = node(0)
        self.head['addresses'][1]['rdma'] = False
        with self.assertRaisesRegex(ValueError, 'RDMA'):
            setup.fabric_pairs(self.head, self.worker, ['192.168.2.0/30', '192.168.2.4/30'])

    def test_management_override_for_jump_host_and_missing_source_revision(self):
        self.worker['ssh_connection'] = '172.16.0.10 1 10.10.0.2 22'
        with self.assertRaisesRegex(ValueError, 'local interface'):
            setup.plan(self.head, self.worker, 'worker')
        data = setup.plan(self.head, self.worker, 'worker', head_ip='10.10.0.1')
        self.assertEqual(data['values'][0]['HEAD_HOST'], '10.10.0.1')
        self.worker['source_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Source differs'):
            setup.plan(self.head, self.worker, 'worker', head_ip='10.10.0.1')

    def test_models_are_resolved_from_ids_in_each_original_cache(self):
        data = setup.plan(self.head, self.worker, 'worker', profile='c6')
        for side, values in enumerate(data['values']):
            self.assertEqual(values['MODEL_ID'], 'nvidia/GLM-5.3-Flash-NVFP4')
            self.assertEqual(values['HF_HUB_HOST_DIR'], f'/cache/user{side}/hub')
            self.assertEqual(values['MODEL_REVISION'], configure.MODEL_REVISION)
            self.assertNotIn('MODEL_DIR', values)
            self.assertEqual(values['MAX_NUM_SEQS'], '6')
            self.assertEqual(values['MAX_NUM_BATCHED_TOKENS'], '6144')
            self.assertEqual(values['KV_CACHE_MEMORY'], '8589934592')
            self.assertEqual(values['DRM_CARD_GID'], str(100 + side))
        self.assertEqual(data['values'][0]['PEER_DEPLOY_DIR'], self.worker['root'])

    def test_hf_cache_environment_precedence(self):
        self.assertEqual(configure.hub_path('/home/u', {}), '/home/u/.cache/huggingface/hub')
        self.assertEqual(configure.hub_path('/home/u', {'XDG_CACHE_HOME': '/xdg'}), '/xdg/huggingface/hub')
        self.assertEqual(configure.hub_path('/home/u', {'HF_HOME': '/hf'}), '/hf/hub')
        self.assertEqual(configure.hub_path('/home/u', {'HF_HOME': '/hf', 'HF_HUB_CACHE': '/models'}), '/models')

    def test_user_local_hf_cli_is_found_even_when_ssh_path_omits_it(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup_probe.shutil, 'which', return_value=None):
            path = Path(tmp) / '.local/bin/hf'
            path.parent.mkdir(parents=True)
            path.write_text('#!/bin/sh\n')
            path.chmod(0o755)
            self.assertEqual(setup_probe.hf_command(tmp), str(path))
            path.chmod(0o644)
            self.assertIsNone(setup_probe.hf_command(tmp))

    def test_invalid_model_id_and_unpinned_revision_fail(self):
        for opts in ({'model_id': '../escape'}, {'model_revision': 'main'}, {'draft_id': 'foo/../bar'}):
            with self.assertRaises(ValueError):
                setup.plan(self.head, self.worker, 'worker', model_options=opts)

    def test_ssh_commands_preserve_payload_and_reject_option_injection(self):
        for target in ('-oProxyCommand=bad', 'a;bad', 'user@x\ny', 'foo bar'):
            with self.assertRaises(ValueError):
                setup.validate_worker(target)
        payload = dict(root="/path with spaces/'quotes'/$HOME/$(touch /tmp/no)")
        command = setup.remote_command('user@worker', 'inventory', payload)
        parsed = shlex.split(command[-1])
        self.assertEqual(json.loads(parsed[-1]), payload)
        self.assertEqual(parsed[:2], ['python3', '-c'])

    def test_firewall_rules_are_exact_peers_both_directions_not_global_policy(self):
        data = setup.plan(self.head, self.worker, 'worker', client_cidr='10.10.0.99')
        for side, rules in enumerate(data['firewall']):
            self.assertEqual(len(rules), 7 if side == 0 else 6)
            for pair, incoming, outgoing in zip(data['pairs'], rules[::2], rules[1::2]):
                self.assertEqual(incoming, ['ufw', 'allow', 'in', 'on', pair[side]['interface'],
                                            'from', pair[1-side]['address'], 'to', pair[side]['address']])
                self.assertEqual(outgoing, ['ufw', 'allow', 'out', 'on', pair[side]['interface'],
                                            'from', pair[side]['address'], 'to', pair[1-side]['address']])
            self.assertNotIn('enable', sum(rules, []))
            self.assertNotIn('reset', sum(rules, []))
            self.assertNotIn('any', sum(rules, []))
        self.assertEqual(data['firewall'][0][-1][-6:], ['from', '10.10.0.99/32', 'to', '10.10.0.1', 'port', '8000'])

    def test_existing_env_and_unknown_ufw_are_reported_not_silently_passed(self):
        self.worker.update(existing_env=True, ufw='unknown')
        data = setup.plan(self.head, self.worker, 'worker')
        self.assertTrue(any('protected' in s for s in data['warnings']))
        self.assertTrue(any('UFW unknown' in s for s in data['warnings']))

    def test_root_only_modeset_is_unknown_without_claiming_disabled(self):
        self.head['checks']['drm_modeset'] = None
        data = setup.plan(self.head, self.worker, 'worker')
        self.assertTrue(any('drm_modeset unknown' in s and 'sudo cat' in s for s in data['warnings']))
        self.assertFalse(any('host preparation needed: drm_modeset' in s for s in data['warnings']))

    def test_missing_download_cli_refuses_asset_preparation_before_config_writes(self):
        self.worker['checks']['tools']['hf'] = False
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(setup, 'ROOT', Path(tmp)), \
                patch.object(setup_probe, 'inventory', return_value=self.head), \
                patch.object(setup, 'request', return_value=self.worker), \
                patch.object(setup, 'apply_plan') as apply, \
                patch.object(sys, 'argv', ['setup.py', '--worker', 'worker', '--apply', '--prepare-assets']), \
                patch('sys.stdout', new_callable=io.StringIO), \
                patch('sys.stderr', new_callable=io.StringIO) as error:
            self.assertEqual(setup.main(), 2)
            self.assertIn('Hugging Face CLI', error.getvalue())
            self.assertIn('No settings were written', error.getvalue())
            apply.assert_not_called()

    def test_private_plan_and_scripts_only_no_model_moves_or_config_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'private plan'
            data = setup.plan(self.head, self.worker, 'worker')
            setup.write_plan(data, directory)
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            for file in directory.iterdir():
                self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600)
            prep = (directory / 'worker.prepare.sh').read_text()
            self.assertIn("export HF_HUB_CACHE=/cache/user1/hub", prep)
            self.assertIn('hf download nvidia/GLM-5.3-Flash-NVFP4 --revision ' + configure.MODEL_REVISION, prep)
            self.assertNotIn('--local-dir', prep)
            fakebin = Path(tmp) / 'bin with spaces'
            fakebin.mkdir()
            trace = Path(tmp) / 'calls'
            for command in ('docker', 'hf'):
                f = fakebin / command
                f.write_text('#!/bin/bash\nprintf "%s " "${0##*/}" "$@" >> "$TRACE"\nprintf "\\n" >> "$TRACE"\n')
                f.chmod(0o755)
            result = subprocess.run(['bash', str(directory / 'worker.prepare.sh')],
                                    env=dict(os.environ, PATH=str(fakebin) + ':' + os.environ['PATH'], TRACE=str(trace)),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(trace.read_text().splitlines()), 3)
            # The discovered executable is used explicitly, including paths with spaces.
            data['nodes'][1]['hf_command'] = str(fakebin / 'hf')
            explicit = Path(tmp) / 'absolute-cli-plan'
            setup.write_plan(data, explicit)
            result = subprocess.run(['bash', str(explicit / 'worker.prepare.sh')],
                                    env=dict(os.environ, PATH=str(fakebin) + ':' + os.environ['PATH'], TRACE=str(trace)),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(trace.read_text().splitlines()), 6)
            for file in directory.glob('*.sh'):
                result = subprocess.run(['bash', '-n', str(file)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            with self.assertRaises(FileExistsError):
                setup.write_plan(data, directory)


class CheckoutProtection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'checkout'
        (self.root / 'manifests').mkdir(parents=True)
        (self.root / 'source').write_text('real source')
        (self.root / 'manifests/source.json').write_text(json.dumps({'files': {'source': hashlib.sha256(b'real source').hexdigest()}}))
        self.expected = setup_probe.source_hash(self.root)
        self.payload = dict(root=str(self.root), source_sha256=self.expected, env='ROLE=head\n', directories=[])

    def test_fresh_apply_uses_exclusive_private_file_and_refuses_repeat(self):
        with patch.object(setup_probe, 'run', return_value=(True, '')):
            setup_probe.apply(self.payload)
            self.assertEqual((self.root / '.env').read_text(), 'ROLE=head\n')
            self.assertEqual(stat.S_IMODE((self.root / '.env').stat().st_mode), 0o600)
            with self.assertRaisesRegex(ValueError, 'protected'):
                setup_probe.apply(self.payload)

    def test_missing_worker_checkout_explains_how_to_prepare_it(self):
        with self.assertRaisesRegex(ValueError, 'clone the same repository'):
            setup_probe.source_hash(self.root / 'missing')

    def test_dangling_env_symlink_and_changed_source_are_protected(self):
        (self.root / '.env').symlink_to('missing')
        with self.assertRaisesRegex(ValueError, 'protected'):
            setup_probe.writable_checkout(self.root, self.expected)
        (self.root / '.env').unlink()
        (self.root / 'source').write_text('modified')
        with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
            setup_probe.writable_checkout(self.root, self.expected)

    def test_live_mount_on_checkout_or_ancestor_is_protected(self):
        for path in (self.root / 'source', self.root, self.root.parent):
            containers = json.dumps([{'Name': 'live', 'Mounts': [{'Source': str(path)}]}])
            with patch.object(setup_probe, 'run', side_effect=[(True, 'id'), (True, containers)]):
                with self.assertRaisesRegex(ValueError, 'running container'):
                    setup_probe.writable_checkout(self.root, self.expected)
        with patch.object(setup_probe, 'run', return_value=(False, 'access denied')):
            with self.assertRaisesRegex(ValueError, 'Docker permissions'):
                setup_probe.writable_checkout(self.root, self.expected)

    def test_remote_apply_is_not_attempted_if_head_is_protected(self):
        (self.root / '.env').write_text('PRIVATE=yes')
        data = setup.plan(node(0), node(1), 'worker')
        data['nodes'][0].update(root=str(self.root), source_sha256=self.expected)
        with patch.object(setup, 'request') as request:
            with self.assertRaisesRegex(ValueError, 'protected'):
                setup.apply_plan(data)
            request.assert_not_called()
        self.assertEqual((self.root / '.env').read_text(), 'PRIVATE=yes')


class TcpProbe(unittest.TestCase):
    def test_bidirectional_loopback_uses_temporary_ports_and_cleans_up(self):
        def local_remote(_worker, mode, payload, code=None):
            return [sys.executable, str(setup.PROBE), mode, json.dumps(payload)]
        pairs = [(dict(address='127.0.0.1'), dict(address='127.0.0.1'))]
        with patch.object(setup, 'remote_command', side_effect=local_remote):
            results = setup.network_checks('worker', pairs)
        self.assertEqual(len(results), 2)
        for result in results:
            port = int(result.rsplit(' ', 1)[-1].rstrip(')'))
            with socket.socket() as sock:
                sock.settimeout(1)
                self.assertNotEqual(sock.connect_ex(('127.0.0.1', port)), 0)

    def test_refused_tcp_connect_is_an_actionable_failure_and_listener_is_closed(self):
        def local_remote(_worker, mode, payload, code=None):
            return [sys.executable, str(setup.PROBE), mode, json.dumps(payload)]
        pairs = [(dict(address='127.0.0.1'), dict(address='127.0.0.1'))]
        with patch.object(setup, 'remote_command', side_effect=local_remote), \
                patch.object(setup_probe, 'tcp_client', side_effect=ConnectionRefusedError('refused')):
            with self.assertRaisesRegex(ValueError, 'peer firewall rules'):
                setup.network_checks('worker', pairs)


if __name__ == '__main__':
    unittest.main()
