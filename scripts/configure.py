#!/usr/bin/env python3
"""Create role-based site settings for a two-node GB10 deployment."""
import argparse
from pathlib import Path
import shlex

p = argparse.ArgumentParser()
p.add_argument('--role', choices=('head', 'worker'), required=True)
p.add_argument('--head-host', required=True)
p.add_argument('--node-ip', required=True)
p.add_argument('--fabric-subnets', required=True, help='Space-separated IPv4 fabric prefixes')
p.add_argument('--image', required=True, help='Installed patched ARM64/GB10 image tag')
p.add_argument('--peer-ssh', default='', help='SSH destination from the head to the worker')
p.add_argument('--peer-dir', default='', help='Deployment directory on the worker')
p.add_argument('--snapshot-seed', default='', help='Optional existing read-only optimized snapshots')
p.add_argument('--drm-gid', default='44')
p.add_argument('--force', action='store_true', help='Explicitly replace an existing .env')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / '.env'
if out.exists() and not a.force:
    p.error('.env exists; use --force only if you intend to replace node settings')
home = Path.home(); hub = home / '.cache/huggingface/hub'; cache = home / '.cache/glm53-native'
log = home / '.local/state/glm53-native/logs'
seed = Path(a.snapshot_seed) if a.snapshot_seed else cache / 'snapshot-seed'
for directory in (cache, log): directory.mkdir(parents=True, exist_ok=True)
if not a.snapshot_seed: seed.mkdir(parents=True, exist_ok=True)
values = dict(ROLE=a.role, NODE_RANK='0' if a.role == 'head' else '1', HEAD_HOST=a.head_host,
    VLLM_HOST_IP=a.node_ip, FABRIC_SUBNETS=a.fabric_subnets, IMAGE=a.image, DEPLOY_ROOT=str(root),
    PEER_SSH=a.peer_ssh, PEER_DEPLOY_DIR=a.peer_dir or str(root), PROJECT_NAME='glm53-native',
    CONTAINER_NAME='glm53-native', MODEL_HOST_DIR=str(hub/'models--nvidia--GLM-5.3-Flash-NVFP4'),
    MODEL_DIR='/models/glm-5.3-flash-nvfp4/snapshots/da920bb0b9f4a06727223a349e55468e38352348',
    DFLASH_HOST_DIR=str(hub/'models--incoai--GLM-5.3-Flash-DFlash2'),
    DFLASH_MODEL='/models/glm-5.3-flash-dflash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a',
    CACHE_HOST_DIR=str(cache), LOG_HOST_DIR=str(log), SNAPSHOT_SEED_DIR=str(seed), TP='2',
    API_PORT='8000', STATUS_PORT='8082', MASTER_PORT='29553', FABRIC_CHECK_PORT='29511',
    MAX_NUM_SEQS='4', MAX_MODEL_LEN='1047552', KV_CACHE_MEMORY='8589934592', MAX_NUM_BATCHED_TOKENS='4096',
    GPU_MEM_UTIL='0.88', TORCH_MEM_FRACTION='0.92', LIMIT_MM='{"image":32,"video":0}',
    MM_PROCESSOR_CACHE_GB='1', SERVED_NAME='glm53', SERVICE_NAME='glm53',
    MODEL_ALIASES='glm53 nvidia/GLM-5.3-Flash-NVFP4', GLM53_DISPLAY_KV_ENABLE='1',
    GLM53_DISPLAY_KV_MIN_BYTES='4294967296', DRM_CARD_GID=a.drm_gid)
out.write_text('# Node settings; no container is started by this command.\n'+''.join(k+'='+shlex.quote(v)+'\n' for k,v in values.items()))
print('Created', out)
