"""Versioned runtime recipes. No host paths or credentials belong in the catalog."""
import json
import os
from pathlib import Path
import re


def catalog(root):
    return json.loads((Path(root) / 'profiles/images.json').read_text())['profiles']


def image_ids(spec):
    # Docker's classic store reports the configuration ID; the containerd
    # store can report the OCI index or platform manifest instead.
    return {spec['index_digest'], spec['platform_digest'], spec['config_digest']}


def selected(root, version=None):
    version = version or os.environ.get('GLM53_PROFILE')
    if version is not None and (not re.fullmatch(r'v\d+\.\d+\.\d+', version)
                                or version not in catalog(root)):
        raise ValueError('Unknown runtime version: ' + str(version))
    return version


def env_file(root, version=None):
    root = Path(root)
    version = selected(root, version)
    private = root / ('.env.' + version) if version else root / '.env'
    return private if private.is_file() else root / '.env'


def compose_files(root, version=None):
    version = selected(root, version)
    return [str(Path(root) / catalog(root)[version]['compose'])] if version else []
