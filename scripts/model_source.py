"""Pinned, offline HF-cache resolution shared by checks and the native loader."""
import copy
import os
from pathlib import Path
import re
import sys

HUB_MOUNT = '/hf-cache/hub'


def cache_snapshot(hub, model_id, revision):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*', model_id or ''):
        raise ValueError('Expected a Hugging Face model ID in namespace/name form')
    if not re.fullmatch(r'[a-f0-9]{40}', revision or ''):
        raise ValueError('Use a pinned 40-character lowercase model commit revision')
    root = Path(hub)
    if not root.is_absolute():
        raise ValueError('HF cache must be an absolute path')
    model = root / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
    if not (model / 'config.json').is_file():
        raise ValueError(f'Pinned model is missing from the offline cache: {model}. Download this revision on both nodes before startup.')
    if not model.resolve().is_relative_to(root.resolve()):
        raise ValueError('Cached model directory escapes the mounted HF cache')
    return model


def compose_command(root, settings):
    root = Path(root)
    mode = 'hf' if settings.get('MODEL_ID') else 'local'
    return ['docker', 'compose', '--env-file', str(root / '.env'), '-p',
            settings.get('PROJECT_NAME', 'glm53-native'), '-f', str(root / 'compose.yaml'),
            '-f', str(root / f'compose.models-{mode}.yaml')]


def snapshot_fingerprint_config(model_config, checkpoint_hash):
    """Identify cached HF weights exactly as the existing local-path snapshots.

    Only normalize when a pinned local snapshot exists and the installed
    metadata hasher recognizes it. Otherwise retain the original model/revision.
    No processed tensor files or snapshot manifests are rewritten.
    """
    hub = Path(os.environ.get('HF_HUB_CACHE', HUB_MOUNT))
    local = Path(model_config.model)
    if local.is_absolute():
        # Offline vLLM may resolve the ID before the weight loader sees it.
        try:
            parts = local.relative_to(hub).parts
        except ValueError:
            return model_config
        if (len(parts) != 3 or not parts[0].startswith('models--') or
                parts[1] != 'snapshots' or parts[2] != model_config.revision or
                not re.fullmatch(r'[a-f0-9]{40}', model_config.revision or '') or
                not (local / 'config.json').is_file() or
                not local.resolve().is_relative_to(hub.resolve())):
            return model_config
    else:
        try:
            local = cache_snapshot(hub, model_config.model, model_config.revision)
        except ValueError:
            return model_config
    if not checkpoint_hash(str(local)):
        return model_config
    result = copy.copy(model_config)
    result.model = str(local)
    # For local checkpoints the metadata fingerprint already identifies weights.
    result.revision = None
    return result


def main():
    draft = len(sys.argv) == 2 and sys.argv[1] == 'draft'
    if len(sys.argv) != 2 or sys.argv[1] not in ('target', 'draft'):
        raise ValueError('Usage: model_source.py target|draft')
    prefix = 'DRAFT' if draft else 'MODEL'
    print(cache_snapshot(os.environ.get('HF_HUB_CACHE', HUB_MOUNT),
                         os.environ.get(prefix + '_ID'), os.environ.get(prefix + '_REVISION')))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))
