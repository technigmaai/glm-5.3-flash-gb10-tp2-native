"""Read-only checks for model bind mounts and the display-KV prerequisites."""
import json
from pathlib import Path, PurePosixPath
import posixpath
import subprocess


def model_host_path(host, mount, container_path):
    """Map a container path under its model mount, including the mount itself."""
    path = PurePosixPath(posixpath.normpath(container_path))
    try:
        relative = path.relative_to(PurePosixPath(mount))
    except ValueError:
        raise ValueError(f'Model path {container_path!r} must be inside {mount}') from None
    return Path(host) / str(relative)


def _accessible_file(path, host_root, label):
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f'{label}: missing or broken model file {path}: {error}') from None
    if not resolved.is_relative_to(host_root):
        raise ValueError(
            f'{label}: {path} resolves outside the model mount to {resolved}. '
            'Mount the complete Hugging Face model cache, including blobs/ and '
            'snapshots/, rather than just snapshots/<revision>.')
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f'{label}: model file is missing, empty or not a file: {path}')


def validate_model_mount(host, mount, container_path, *, tokenizer=False, label='Model'):
    """Check metadata, symlink containment and shard presence without reading weights."""
    host_root = Path(host).resolve(strict=True)
    model = model_host_path(host, mount, container_path)
    if not model.is_dir():
        raise ValueError(f'{label}: model directory does not exist: {model}')
    if not model.resolve().is_relative_to(host_root):
        raise ValueError(f'{label}: model directory resolves outside its bind mount: {model}')
    for path in model.rglob('*'):
        if path.is_symlink():
            # An absolute host link will not be relocated by a Docker bind mount.
            if path.readlink().is_absolute():
                raise ValueError(f'{label}: absolute symlink {path} is not portable inside '
                                 'the model mount; use a complete cache with relative links.')
            _accessible_file(path, host_root, label)
    required = ['config.json']
    if tokenizer:
        required += ['tokenizer.json', 'tokenizer_config.json']
    for name in required:
        _accessible_file(model / name, host_root, label)
    config = json.loads((model / 'config.json').read_text())
    indexes = list(model.glob('*.safetensors.index.json'))
    if indexes:
        for index in indexes:
            _accessible_file(index, host_root, label)
            shards = set(json.loads(index.read_text()).get('weight_map', {}).values())
            if not shards:
                raise ValueError(f'{label}: empty weight map in {index}')
            for shard in shards:
                path = model_host_path(host, mount, f'{container_path}/{shard}')
                _accessible_file(path, host_root, label)
    else:
        shards = list(model.glob('*.safetensors'))
        if not shards:
            raise ValueError(f'{label}: no safetensors weights or index under {model}')
        for path in shards:
            _accessible_file(path, host_root, label)
    return config


def display_kv_checks(settings, modeset_file=Path('/sys/module/nvidia_drm/parameters/modeset')):
    """Return errors and advisories; unreadable root-only sysfs is not a failed mode."""
    if settings.get('GLM53_DISPLAY_KV_ENABLE', '1') != '1':
        return [], []
    try:
        modeset = Path(modeset_file).read_text().strip().upper()
    except PermissionError:
        # Some DGX OS images expose this parameter as root:root mode 0400.
        # Try only a non-interactive read; never prompt for a password or change it.
        try:
            result = subprocess.run(['sudo', '-n', 'cat', str(modeset_file)],
                                    capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is None or result.returncode != 0:
            return [], [f'DRM modesetting could not be verified: {modeset_file} is root-only. '
                        f'Run sudo cat {modeset_file} on this host and confirm Y/1 before '
                        'starting display KV. See docs/DEPLOYMENT.md#drm-modesetting.']
        modeset = result.stdout.strip().upper()
    except OSError as error:
        return [f'Display KV requires nvidia_drm modeset=1; cannot read {modeset_file}: {error}. '
                'See docs/DEPLOYMENT.md#drm-modesetting.'], []
    if modeset not in ('Y', '1'):
        return [f'Display KV requires nvidia_drm modeset=1, but {modeset_file} is {modeset!r}. '
                'See docs/DEPLOYMENT.md#drm-modesetting.'], []
    return [], []
