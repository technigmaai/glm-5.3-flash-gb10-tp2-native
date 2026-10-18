# Technigma AI — GLM-5.3-Flash on two GB10 nodes

**Versioned Linux ARM64 runtime images for native two-node TP2.** One model
container per GB10 node; OpenAI-compatible API on the head at port 8000.

**Use the [GitHub repository](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native) and its single launcher.** These images
require the repository's entrypoint and matching source overlays. Pulling an
image alone does not install the deployment or download model weights.

[Installation](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/docs/DEPLOYMENT.md) ·
[Release contents](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/VERSIONS.md) ·
[Launcher guide](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/launchers/README.md) ·
[Issues](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/issues)

## Choose a version

Image name: `technigmaai/glm-5.3-flash-gb10-tp2-native:<version>`

| Tag | Main change | Status |
|---|---|---|
| `v0.1.0` | Original native TP2 runtime | Original baseline |
| `v0.2.0` | ARX synchronization fix | Patched baseline |
| `v0.3.0` | Collective counters, tool streaming, attention and draft graphs | Experiment |
| `v0.4.0` | MLA prefill workspace lifetime | Experiment |
| `v0.5.0` | Relay handling, block-table cleanup and sampler | Experiment |
| `v0.6.0` | Speculative randomness and verification | Experiment; operator's running image observed 2026-10-18 |

Versions are consecutive release identifiers, independent of patch counts.
Use v0.1.0 to reproduce the original baseline; v0.6.0 identifies the operator's
observed running image. This is not a performance ranking or production-readiness
claim. Historical long-form tags remain compatibility references; use the version
tags for new deployments. Exact changes, upstream pins and validation limits are
in [VERSIONS.md](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/VERSIONS.md).

Example pull (run on **both** prepared nodes):

```bash
docker pull technigmaai/glm-5.3-flash-gb10-tp2-native:v0.1.0
```

## Requirements

Two headless Linux ARM64 GB10 systems, such as NVIDIA DGX Spark or ASUS GX10,
with Docker Engine, Compose v2, NVIDIA Container Toolkit, compatible drivers,
working RoCE, DRM access for display-backed KV, Python 3 and passwordless SSH
from head to worker. Prepare the required host tools and model access as described
in the [deployment guide](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/docs/DEPLOYMENT.md).

Models are cached separately on each node and mounted read-only:

| Model | Pinned revision |
|---|---|
| `nvidia/GLM-5.3-Flash-NVFP4` | `da920bb0b9f4a06727223a349e55468e38352348` |
| `incoai/GLM-5.3-Flash-DFlash2` | `bf582e4eacc1810f76656d1811693ff6c6737d2a` |

## Install and launch

Clone the same Git revision on **both** nodes:

```bash
git clone https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native.git
cd glm-5.3-flash-gb10-tp2-native
```

On the **head**, review a fresh setup plan:

```bash
./launchers/launch.sh setup --worker user@worker --version v0.1.0
```

Replace `user@worker`. Add `--worker-dir /absolute/worker/checkout` if the worker
uses a different checkout path. Review host warnings and any firewall plan, then
create the separate node settings and prepare the selected image/models:

```bash
./launchers/launch.sh setup --worker user@worker --version v0.1.0 \
  --apply --prepare-assets --check-network
```

Existing `.env` files are protected. Setup does not start containers or apply
firewall rules. Use the [migration guide](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/docs/LAYOUT_MIGRATION.md)
for an existing deployment.

Open the **single terminal launcher**:

```bash
./launchers/launch.sh
```

Its Technigma AI menu uses Up/Down and Enter for image selection, start/switch,
status, restart, stop, setup, checks and logs. It works over SSH without a GUI.

For non-interactive operation:

```bash
./launchers/launch.sh check --version v0.1.0
./launchers/launch.sh start --version v0.1.0 --dry-run
./launchers/launch.sh start --version v0.1.0 --approved
./launchers/launch.sh status
./launchers/launch.sh logs --node worker --tail 80
./launchers/launch.sh stop --approved
```

To switch, pull the other version on both nodes, then select it in the head's menu
or pass `start --version v0.6.0 --approved`. Both nodes are checked before a switch
interrupts the pair. Startup is asynchronous; inspect Status, Logs and the head's
`/health` endpoint. Switching does not build images or rewrite private settings.

## Serving settings and validation

Image versions select code. C4/C6 select serving settings; they are separate.
Fresh setup defaults to **C4: 4,096 batched tokens, four slots, 8 GiB logical KV
per GPU**. C6 uses 6,144 batched tokens and six slots with the same 8 GiB pool.
The 1.75 GiB display-backed allocation is part of the pool, not additional capacity.
The configured context limit is 1,047,552 tokens; capacity is shared across requests.

The operator's v0.6.0 observation used 16 slots / 7 GiB KV. That observation does
not impose those settings on a fresh installation or qualify every workload.
Original runtime benchmarks, later activation smoke checks and CPU/configuration
tests have different scopes. See [validation details](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/VERSIONS.md#validation)
before interpreting them. Automatic generation self-tests are disabled; explicit
verification submits real model requests. No recovery watchdog is installed.

## Source, digests and licenses

The common stack is CUDA 13.0, PyTorch 2.13.0+cu130, pinned vLLM 0.30.1rc1.dev193,
FlashInfer 0.7.0 and Triton 3.7.1, with scoped version-specific backports.
The recipe retains DFlash2, adaptive-k, RecoverSSM and weight snapshots and uses
native ranks in place of Mentat/Ray services.

[Digest catalog](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/profiles/images.json) ·
[Source provenance](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/tree/main/profiles/provenance) ·
[License notices](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/THIRD_PARTY.md)

Derived from Kindling AI's serving stack and the coolbho3k display-memory technique.
Source licenses and target/draft model terms remain separate; publication does
not change their conditions. The [GitHub documentation](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native/blob/main/docs/README.md)
is the canonical installation and maintenance reference.
