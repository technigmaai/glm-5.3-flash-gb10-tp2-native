"""CPU-only regression checks; no Docker, GPU, network or host changes."""
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment_checks import display_kv_checks, model_host_path, validate_model_mount
import check

MOUNT = '/models/glm-5.3-flash-nvfp4'
DRAFT = '/models/glm-5.3-flash-dflash2'


def flat(root, tokenizer=True):
    root.mkdir(parents=True)
    (root / 'config.json').write_text('{"max_position_embeddings":1048576}')
    (root / 'model.safetensors').write_bytes(b'fixture')
    if tokenizer:
        (root / 'tokenizer.json').write_text('{}')
        (root / 'tokenizer_config.json').write_text('{}')
    return root


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.host = flat(self.root / 'model')

    def validate(self, path=MOUNT):
        return validate_model_mount(self.host, MOUNT, path, tokenizer=True)

    def cache(self):
        snapshot = self.host / 'snapshots/revision'
        snapshot.mkdir(parents=True)
        blobs = self.host / 'blobs'
        blobs.mkdir()
        for path in list(self.host.iterdir()):
            if path.is_file():
                path.rename(blobs / path.name)
                (snapshot / path.name).symlink_to('../../blobs/' + path.name)
        return snapshot

    def test_bare_and_dot_mount_map_to_host_root(self):
        for path in (MOUNT, MOUNT + '/', MOUNT + '/.'):
            with self.subTest(path=path):
                self.assertEqual(model_host_path(self.host, MOUNT, path), self.host)
                self.assertEqual(self.validate(path)['max_position_embeddings'], 1048576)

    def test_complete_hf_cache_links_resolve(self):
        self.cache()
        self.validate(MOUNT + '/snapshots/revision')

    def test_snapshot_only_mount_is_rejected_even_if_host_links_work(self):
        snapshot = self.cache()
        self.assertTrue((snapshot / 'tokenizer.json').is_file())
        with self.assertRaisesRegex(ValueError, 'complete Hugging Face model cache'):
            validate_model_mount(snapshot, MOUNT, MOUNT, tokenizer=True)

    def test_broken_tokenizer_link_is_rejected(self):
        snapshot = self.cache()
        (self.host / 'blobs/tokenizer.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing or broken model file'):
            self.validate(MOUNT + '/snapshots/revision')

    def test_absolute_host_link_is_rejected(self):
        p = self.host / 'tokenizer.json'
        p.unlink()
        blob = self.host / 'absolute.json'
        blob.write_text('{}')
        p.symlink_to(blob)
        with self.assertRaisesRegex(ValueError, 'absolute symlink'):
            self.validate()

    def test_missing_and_empty_tokenizer_are_rejected(self):
        p = self.host / 'tokenizer.json'
        p.unlink()
        with self.assertRaisesRegex(ValueError, 'tokenizer.json'):
            self.validate()
        p.touch()
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.validate()

    def test_path_outside_model_mount_is_rejected(self):
        for path in ('/models/other', MOUNT + '/../../outside', 'relative/path'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'must be inside'):
                model_host_path(self.host, MOUNT, path)

    def test_symlinked_model_directory_cannot_escape_mount(self):
        outside = flat(self.root / 'outside')
        (self.host / 'escaped').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'directory resolves outside'):
            self.validate(MOUNT + '/escaped')

    def test_indexed_missing_shard_is_rejected(self):
        index = self.host / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a': 'model.safetensors', 'b': 'missing.safetensors'}}))
        with self.assertRaisesRegex(ValueError, 'missing.safetensors'):
            self.validate()
        (self.host / 'missing.safetensors').write_bytes(b'fixture')
        self.validate()

    def test_no_weights_is_rejected(self):
        (self.host / 'model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'no safetensors'):
            self.validate()

    def test_drafter_needs_weights_and_config_but_not_tokenizer(self):
        draft = flat(self.root / 'draft', tokenizer=False)
        validate_model_mount(draft, DRAFT, DRAFT, label='DFLASH_MODEL')

    def run_check_cli(self, earlyoom_active=False):
        draft = flat(self.root / 'draft', tokenizer=False)
        env = dict(ROLE='head', NODE_RANK='0', TP='2', MAX_MODEL_LEN='1047552',
                   MAX_NUM_SEQS='4', KV_CACHE_MEMORY='8589934592', LIMIT_MM='{}',
                   MODEL_DIR=MOUNT, DFLASH_MODEL=DRAFT, GLM53_DISPLAY_KV_ENABLE='0')
        rendered = {'services': {'glm53': {'image': 'fixture', 'environment': env,
                    'volumes': [{'source': str(self.host), 'target': MOUNT, 'read_only': True},
                                {'source': str(draft), 'target': DRAFT, 'read_only': True}]}}}
        with patch.object(check, 'load_settings', return_value={}), \
             patch.object(check.subprocess, 'check_output', return_value=json.dumps(rendered)), \
             patch.object(check.subprocess, 'run', side_effect=lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0 if cmd[0] != 'systemctl' or earlyoom_active else 3)), \
             patch('sys.stderr', new_callable=io.StringIO) as stderr:
            check.main()
            return stderr.getvalue()

    def test_check_cli_accepts_flat_target_and_drafter(self):
        self.assertEqual(self.run_check_cli(), '')

    def test_active_earlyoom_warns_without_blocking_or_stopping_service(self):
        self.assertIn('WARN: earlyoom is active', self.run_check_cli(earlyoom_active=True))


class DisplayChecks(unittest.TestCase):
    def test_enabled_modeset_and_failure_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'modeset'
            for value in ('Y\n', '1\n'):
                p.write_text(value)
                self.assertEqual(display_kv_checks({}, p), ([], []))
            p.write_text('N\n')
            self.assertIn('modeset=1', display_kv_checks({}, p)[0][0])
            p.unlink()
            self.assertIn('cannot read', display_kv_checks({}, p)[0][0])

    def test_disabled_display_kv_does_not_require_modeset(self):
        self.assertEqual(display_kv_checks({'GLM53_DISPLAY_KV_ENABLE': '0'}, '/missing/modeset'), ([], []))

    def test_root_only_parameter_read_with_noninteractive_sudo(self):
        with patch.object(Path, 'read_text', side_effect=PermissionError()), \
             patch('deployment_checks.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'Y\n', '')) as run:
            self.assertEqual(display_kv_checks({}), ([], []))
            self.assertEqual(run.call_args.args[0][:3], ['sudo', '-n', 'cat'])

    def test_root_only_parameter_without_sudo_warns_instead_of_rejecting(self):
        with patch.object(Path, 'read_text', side_effect=PermissionError()), \
             patch('deployment_checks.subprocess.run', return_value=subprocess.CompletedProcess([], 1, '', 'password required')):
            errors, advisories = display_kv_checks({})
            self.assertEqual(errors, [])
            self.assertIn('could not be verified', advisories[0])

    def test_sudo_read_confirms_disabled_modeset_is_an_error(self):
        with patch.object(Path, 'read_text', side_effect=PermissionError()), \
             patch('deployment_checks.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'N\n', '')):
            errors, advisories = display_kv_checks({})
            self.assertIn('modeset=1', errors[0])
            self.assertEqual(advisories, [])


if __name__ == '__main__':
    unittest.main()
