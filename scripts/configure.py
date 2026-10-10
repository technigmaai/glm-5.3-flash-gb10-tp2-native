#!/usr/bin/env python3
"""Create role-based site settings for a two-node GB10 deployment."""
import argparse
import os
import re
from pathlib import Path
import shlex
from fabric_selectors import parse_selectors

IMAGE = 'technigmaai/glm-5.3-flash-gb10-tp2-native:c748079-displaykv1-arm64-cu130'

MODEL_ID = 'nvidia/GLM-5.3-Flash-NVFP4'
MODEL_REVISION = 'da920bb0b9f4a06727223a349e55468e38352348'
DRAFT_ID = 'incoai/GLM-5.3-Flash-DFlash2'
DRAFT_REVISION = 'bf582e4eacc1810f76656d1811693ff6c6737d2a'


def hub_path(home, env):
    base = env.get('HF_HOME') or str(Path(env.get('XDG_CACHE_HOME', str(Path(home) / '.cache'))) / 'huggingface')
    return env.get('HF_HUB_CACHE') or env.get('HUGGINGFACE_HUB_CACHE') or str(Path(base) / 'hub')


def build_values(*, root, home, role, head_host, node_ip, fabric_subnets,
                 image=IMAGE, peer_ssh='', peer_dir='', snapshot_seed='',
                 drm_gid='44', hub=None, profile='c4', model_id=MODEL_ID, model_revision=MODEL_REVISION,
                 draft_id=DRAFT_ID, draft_revision=DRAFT_REVISION):
    parse_selectors(fabric_subnets.split())
    if role not in ('head', 'worker') or profile not in ('c4', 'c6'):
        raise ValueError('Expected head/worker role and c4/c6 profile')
    for repo, revision in ((model_id, model_revision), (draft_id, draft_revision)):
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*', repo):
            raise ValueError('Expected a Hugging Face model ID in namespace/name form')
        if not re.fullmatch(r'[a-f0-9]{40}', revision):
            raise ValueError('Use a pinned 40-character lowercase model commit revision')
    root, home = Path(root), Path(home)
    hub = Path(hub) if hub else home / '.cache/huggingface/hub'
    cache = home / '.cache/glm53-native'
    log = home / '.local/state/glm53-native/logs'
    seed = Path(snapshot_seed) if snapshot_seed else cache / 'snapshot-seed'
    return dict(ROLE=role, NODE_RANK='0' if role == 'head' else '1', HEAD_HOST=head_host,
        VLLM_HOST_IP=node_ip, FABRIC_SUBNETS=fabric_subnets, IMAGE=image, DEPLOY_ROOT=str(root),
        PEER_SSH=peer_ssh, PEER_DEPLOY_DIR=peer_dir or str(root), PROJECT_NAME='glm53-native',
        CONTAINER_NAME='glm53-native', MODEL_ID=model_id, MODEL_REVISION=model_revision, DRAFT_ID=draft_id, DRAFT_REVISION=draft_revision,
        HF_HUB_HOST_DIR=str(hub),
        CACHE_HOST_DIR=str(cache), LOG_HOST_DIR=str(log), SNAPSHOT_SEED_DIR=str(seed), TP='2',
        API_PORT='8000', STATUS_PORT='8082', MASTER_PORT='29553', FABRIC_CHECK_PORT='29511',
        MAX_NUM_SEQS='6' if profile == 'c6' else '4', MAX_MODEL_LEN='1047552', KV_CACHE_MEMORY='8589934592', MAX_NUM_BATCHED_TOKENS='6144' if profile == 'c6' else '4096',
        GPU_MEM_UTIL='0.88', TORCH_MEM_FRACTION='0.92', LIMIT_MM='{"image":32,"video":0}',
        MM_PROCESSOR_CACHE_GB='1', SERVED_NAME='glm53', SERVICE_NAME='glm53',
        MODEL_ALIASES='glm53 nvidia/GLM-5.3-Flash-NVFP4', GLM53_DISPLAY_KV_ENABLE='1',
        GLM53_DISPLAY_KV_MIN_BYTES='4294967296', DRM_CARD_GID=drm_gid)


def render(values):
    return '# Node settings; no container is started by this command.\n' + ''.join(
        k + '=' + shlex.quote(str(v)) + '\n' for k, v in values.items())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=('head', 'worker'), required=True)
    p.add_argument('--head-host', required=True)
    p.add_argument('--node-ip', required=True)
    p.add_argument('--fabric-subnets', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--peer-ssh', default='')
    p.add_argument('--peer-dir', default='')
    p.add_argument('--snapshot-seed', default='')
    p.add_argument('--drm-gid', default='44')
    p.add_argument('--profile', choices=('c4', 'c6'), default='c4')
    p.add_argument('--model-id', default=MODEL_ID)
    p.add_argument('--model-revision', default=MODEL_REVISION)
    p.add_argument('--draft-id', default=DRAFT_ID)
    p.add_argument('--draft-revision', default=DRAFT_REVISION)
    p.add_argument('--force', action='store_true', help='Explicitly replace an existing .env')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    home = Path.home()
    hub = hub_path(home, os.environ)
    try:
        values = build_values(root=root, home=home, hub=hub, **{k: v for k, v in vars(a).items() if k != 'force'})
        out = root / '.env'
        # O_EXCL also rejects dangling symlinks; --force remains a manual-only escape hatch.
        flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if a.force else os.O_EXCL)
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        with os.fdopen(os.open(out, flags, 0o600), 'w') as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(render(values))
        for key in ('CACHE_HOST_DIR', 'LOG_HOST_DIR', 'SNAPSHOT_SEED_DIR'):
            Path(values[key]).mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as error:
        p.error(str(error))
    print('Created', out)


if __name__ == '__main__':
    main()
