"""Offline model identity and processed-snapshot compatibility regressions."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from model_source import cache_snapshot, compose_command, snapshot_fingerprint_config
from deployment_checks import validate_model_mount

REV = 'a' * 40


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.hub = Path(self.temp.name)
        self.snapshot = self.hub / 'models--org--model' / 'snapshots' / REV
        self.snapshot.mkdir(parents=True)
        blobs = self.hub / 'models--org--model' / 'blobs'
        blobs.mkdir()
        for name, data in [('config.json', b'{}'), ('model.safetensors', b'fixture'),
                           ('tokenizer.json', b'{}'), ('tokenizer_config.json', b'{}')]:
            (blobs / name).write_bytes(data)
            (self.snapshot / name).symlink_to('../../blobs/' + name)

    def test_complete_pinned_snapshot_works_without_refs_or_network(self):
        local = cache_snapshot(self.hub, 'org/model', REV)
        self.assertEqual(local, self.snapshot)
        validate_model_mount(self.hub, '/hf-cache/hub',
                             '/hf-cache/hub/' + str(local.relative_to(self.hub)), tokenizer=True)

    def test_missing_revision_and_unpinned_or_escaping_inputs_fail(self):
        for model, rev in [('org/model', 'b' * 40), ('org/model', 'main'),
                           ('../escape', REV), ('org/model', '../escape')]:
            with self.subTest(model=model, rev=rev), self.assertRaises(ValueError):
                cache_snapshot(self.hub, model, rev)

    def test_broken_blob_is_detected_before_loading(self):
        (self.hub / 'models--org--model/blobs/tokenizer.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing or broken'):
            validate_model_mount(self.hub, '/hf-cache/hub',
                                 '/hf-cache/hub/' + str(self.snapshot.relative_to(self.hub)), tokenizer=True)

    def test_same_cached_checkpoint_can_reuse_local_snapshot_identity(self):
        model = SimpleNamespace(model='org/model', revision=REV, dtype='bf16')
        with patch.dict(os.environ, HF_HUB_CACHE=str(self.hub)):
            mapped = snapshot_fingerprint_config(model, lambda p: 'known-checkpoint')
        self.assertEqual(mapped.model, str(self.snapshot))
        self.assertIsNone(mapped.revision)
        self.assertEqual(mapped.dtype, model.dtype)
        self.assertEqual(model.model, 'org/model')
        self.assertEqual(model.revision, REV)

    def test_unknown_checkpoint_hash_does_not_reuse_old_identity(self):
        model = SimpleNamespace(model='org/model', revision=REV)
        with patch.dict(os.environ, HF_HUB_CACHE=str(self.hub)):
            self.assertIs(snapshot_fingerprint_config(model, lambda p: None), model)

    def test_missing_or_local_checkpoint_preserves_original_fingerprint(self):
        for model in [SimpleNamespace(model='org/model', revision='b' * 40),
                      SimpleNamespace(model='/models/flat', revision=None)]:
            with patch.dict(os.environ, HF_HUB_CACHE=str(self.hub)):
                self.assertIs(snapshot_fingerprint_config(model, lambda p: 'known'), model)

    def test_mount_mode_follows_model_id_not_deployment_hostname(self):
        self.assertIn('/repo/compose.models-hf.yaml', compose_command('/repo', {'MODEL_ID': 'org/model'}))
        self.assertIn('/repo/compose.models-local.yaml', compose_command('/repo', {}))


if __name__ == '__main__':
    unittest.main()
