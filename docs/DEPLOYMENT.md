# Install and operate the native GB10 pair

Use the same repository revision on both nodes. Run the launcher from the head;
it coordinates the worker over SSH. Choose the image version independently of
serving settings such as C4/C6, concurrency and KV memory.

## 1. Prepare both hosts

Both nodes need Linux ARM64, a GB10 GPU, Docker Engine with Compose v2, NVIDIA
Container Toolkit, compatible GPU drivers, Python 3 and the Hugging Face CLI.
Configure passwordless SSH from the head to the worker and working RoCE links.

The native display-memory recipe requires headless nodes, NVIDIA DRM modesetting
and access to `/dev/dri/card0` and `/dev/infiniband`. Normal preflight checks two
RoCE interfaces at MTU 9000, an idle GPU, free serving/rendezvous ports and at
least 100 GiB available host RAM. Prepare the hosts before loading the model;
the launcher does not change display services or firewall policy for you.

For detailed host diagnostics, see the [original runtime reference](REFERENCE.md#troubleshooting-first-boot).
Image layers and the two model caches require disk space on each node. Models
are downloaded separately and mounted read-only.

## 2. Clone the source on both nodes

```bash
git clone https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native.git
cd glm-5.3-flash-gb10-tp2-native
git rev-parse HEAD
```

Both commits must match. The checkout paths may differ; give setup the worker's
absolute path with `--worker-dir` when it differs from the head path.

## 3. Choose an image version

Read [VERSIONS.md](../VERSIONS.md). The commands below use **v0.1.0**, the original
baseline. To install another release, use that same version in setup, Check and
Start. **v0.6.0** is the operator's running image observed on 2026-10-18; its
experimental status and recorded limits are documented in the version catalog.

Use `./launchers/launch.sh` for the interactive menu, or the commands below.
The image alone does not contain this deployment's complete source recipe.

## 4. Review a fresh setup plan

On the **head**, replace `user@worker` with your worker's SSH destination:

```bash
./launchers/launch.sh setup --worker user@worker --version v0.1.0
```

Setup discovers addresses, fabric interfaces, each user's HF cache and the DRM
group, then saves a private plan under `site/`. Review its node settings and
warnings. If a firewall is enabled or its state is unknown, review and apply the
peer-scoped rules deliberately before checking network reachability. Setup
never applies those firewall rules automatically.

Existing `.env` files are protected: fresh setup refuses to overwrite them.
For an existing deployment, use [the migration guide](LAYOUT_MIGRATION.md).

## 5. Create settings and prepare assets

Once host preparation and any firewall rules are ready, run on the head:

```bash
./launchers/launch.sh setup --worker user@worker --version v0.1.0 \
  --apply --prepare-assets --check-network
```

Add the same `--worker-dir /absolute/worker/checkout` to both setup commands if
needed. This creates a separate private `.env` on each node, pulls the selected
image on both nodes and downloads or reuses the pinned models in each user's
existing Hugging Face cache. It does not start model containers.

| Model | Pinned revision |
|---|---|
| `nvidia/GLM-5.3-Flash-NVFP4` | `da920bb0b9f4a06727223a349e55468e38352348` |
| `incoai/GLM-5.3-Flash-DFlash2` | `bf582e4eacc1810f76656d1811693ff6c6737d2a` |

Do not copy the head's `.env` onto the worker. `DEPLOY_ROOT` must identify each
node's own checkout; `PEER_DEPLOY_DIR` identifies the worker's checkout.
Model, cache, log and snapshot paths remain node-local. For manual configuration,
start from [`.env.example`](../.env.example) and the [host/model reference](REFERENCE.md#deployment).

## 6. Check and start

On the head:

```bash
./launchers/launch.sh check --version v0.1.0
./launchers/launch.sh start --version v0.1.0 --dry-run
./launchers/launch.sh start --version v0.1.0 --approved
./launchers/launch.sh status
```

Both nodes' images, configuration, source and cached models are checked before a
switch can stop either rank. Once the previous pair is stopped, both nodes must
pass idle-resource and fabric checks before either new rank starts.

Startup continues after the command returns. Initial loading/conversion and
kernel compilation can take longer than a cached restart. Use Status and Logs,
then check the **head address**:

```bash
curl --fail http://HEAD_IP:8000/health
curl --fail http://HEAD_IP:8000/v1/models
```

Replace `HEAD_IP` with the head's reachable IPv4 address. `/v1/models` lists
aliases for the same loaded checkpoint. Automatic generation self-tests are
disabled by versioned recipes; health and model metadata do not constitute a
quality or performance test.

## Daily operations

Open `./launchers/launch.sh` to use the menu, or run:

```bash
./launchers/launch.sh status --json
./launchers/launch.sh logs --node head --tail 80
./launchers/launch.sh logs --node worker --tail 80
./launchers/launch.sh restart --approved
./launchers/launch.sh stop --approved
```

Restart and Stop select the recognized running version. A switch interrupts
in-flight requests. Unknown deployments and mixed-version pairs are refused.
If an operation fails partway through, inspect both ranks before retrying;
there is no automatic rollback or restart watchdog.

`./launchers/launch.sh verify --version v0.1.0` submits real inference requests
(alias, image, tool and short concurrency checks). Run it when that extra model
load is appropriate. It does not qualify arbitrary million-token workloads.

## Switch versions

Pull the chosen tag on **both nodes**, then choose it in the head's menu:

```bash
# On BOTH nodes:
docker pull technigmaai/glm-5.3-flash-gb10-tp2-native:v0.6.0

# On the HEAD:
./launchers/launch.sh check --version v0.6.0
./launchers/launch.sh start --version v0.6.0 --dry-run
./launchers/launch.sh start --version v0.6.0 --approved
```

Each release selects its image, source overlays and extension-cache namespace.
Image downloads/builds never happen implicitly during a switch. Version tags
and digests are recorded in [profiles/images.json](../profiles/images.json).
Rollback uses the same steps with a previously installed version.

## Serving settings

The **image version** selects code. The **serving profile** selects memory and
batching settings; it does not select an image.

| Fresh setup choice | Batched tokens | Sequence slots | Logical KV per GPU |
|---|---:|---:|---:|
| C4 (default) | 4096 | 4 | 8 GiB |
| C6 (`setup --serving-profile c6`) | 6144 | 6 | 8 GiB |

Both configure a 1,047,552-token context limit and image input. The 1.75 GiB
display-backed allocation is included within the KV pool. Capacity is shared
across requests; sequence slots are not private million-token allowances.
Recorded original-runtime results are linked in [VERSIONS.md](../VERSIONS.md#validation).

Keep serving settings consistent on both nodes. Change each node's private
configuration and deliberately restart the pair to apply changes. Changing
batch shapes can require a new optimized snapshot and a slower first load.
The operator's recorded v0.6.0 settings (16 slots / 7 GiB KV) are an observation,
not the fresh-install default or a universal capacity recommendation.

### Configuration files for image versions

**A separate `.env` for every image version is optional.** Each node needs its
own base `.env` in the repository root. By default, all image versions share
that node's base settings; choosing a version selects its image and matching
source overlays automatically.

For a selected version, the launcher reads files in this order:

| Selected version | File used on each node |
|---|---|
| `v0.6.0`, with `.env.v0.6.0` present | That node's `.env.v0.6.0` |
| `v0.6.0`, without `.env.v0.6.0` | That node's `.env` |
| Any other version | Its `.env.<version>` if present, otherwise `.env` |

To keep different serving settings for v0.6.0, run these commands from the
repository root **on both nodes**, using each node's own existing `.env`:

```bash
cp -p .env .env.v0.6.0
chmod 600 .env.v0.6.0
${EDITOR:-nano} .env.v0.6.0
```

Skip the copy if the version file already exists. A version file must be a
**complete configuration**, not a partial override. For example, change
`MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS` or `KV_CACHE_MEMORY` consistently on
both nodes. Keep each node's `ROLE`, `NODE_RANK` and network settings from its
own base file; do not copy the head's configuration onto the worker. Node
identity, project, container name, peer path and lifecycle lock/log directory
must remain consistent with the base file.

On the head, validate both nodes before applying the settings:

```bash
./launchers/launch.sh check --version v0.6.0
./launchers/launch.sh start --version v0.6.0 --dry-run
./launchers/launch.sh restart --version v0.6.0 --approved
```

Restart interrupts both ranks and applies changes even when v0.6.0 is already
running. Use `start --version v0.6.0 --approved` when switching from another
version. To return a version to shared settings, remove or move aside its
`.env.<version>` file on both nodes, check again, then restart deliberately.

Private `.env` and `.env.<version>` files are ignored by Git. The launcher never
creates the optional version files or edits `.env` to switch images. Original
v0.1.0 forces ARX two-shot off; patched versions read the explicit per-node
setting and default to 2048 KiB when it is omitted.

## Troubleshooting

| Symptom | First step |
|---|---|
| Image missing or identity mismatch | Pull the exact version tag on both nodes; check the pinned digests. |
| Source/configuration differs | Check out the same commit on both nodes; inspect each private `.env`. |
| API unavailable after Start | Inspect both logs and Status; loading/warmup may still be running. |
| Idle-resource or fabric preflight fails | Read the reported GPU, memory, listener, interface or GID problem before retrying. |
| Existing `.env` blocks Setup | Preserve it and follow the migration guide; fresh setup is not an overwrite command. |
| Menu unavailable through a non-interactive SSH command | Use an SSH terminal (`ssh -t`) or the explicit command-line arguments. |

More detail: [reference troubleshooting](REFERENCE.md#troubleshooting-first-boot),
[manual host/model setup](REFERENCE.md#deployment), [migration](LAYOUT_MIGRATION.md)
and [source maintenance](../image/README.md).
