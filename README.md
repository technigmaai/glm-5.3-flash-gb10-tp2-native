# GLM-5.3-Flash on 2× NVIDIA GB10 — native TP2

Run GLM-5.3-Flash NVFP4 across two GB10 systems, including NVIDIA DGX Spark
and ASUS GX10, with **one container per node** and an OpenAI-compatible API
on **port 8000**.

[Docker Hub](https://hub.docker.com/r/technigmaai/glm-5.3-flash-gb10-tp2-native)
· [Full setup guide](docs/DEPLOYMENT.md#deployment)
· [Validation summary](manifests/validation-summary.json)

Derived from [Kindling AI's GLM recipe](https://github.com/kindlingai/glm-5.3-flash-gx10/tree/c748079d45e6e070b2acb108a91edfe52f4a7747).
It retains the GB10 optimizations, DFlash2 speculation, adaptive-k, RecoverSSM
and weight snapshots, uses native vLLM multiprocessing, and adds display-backed
KV memory for headless hosts.

## Architecture

**Two containers total:**

| Node | Container | Role |
|---|---|---|
| Head | `glm53-native` | API on port 8000 and TP rank 0 |
| Worker | `glm53-native` | Headless TP rank 1 |

TP2 communication runs over RoCE. There is no running Mentat or Ray service.
The status helper runs inside each model container. Models stay in each user's
Hugging Face cache and are mounted read-only; weights are not included in the
image.

## Serving options

Both profiles use a **1,047,552-token context limit**, **8 GiB logical KV per
GPU**, and **image input**. The display-memory patch backs 1.75 GiB within each
KV pool.

| Profile | Batched tokens | Sequence slots | Default KV per GPU |
|---|---:|---:|---:|
| **4,096 / C4** | 4,096 | 4 | 8 GiB |
| **6,144 / C6** | 6,144 | 6 | 8 GiB |

C6 is the current installation's active profile; C4 is the conservative
fresh-install default. Set these values identically in **both nodes' `.env`**:

```dotenv
# C4: 4096 and 4; C6: 6144 and 6
MAX_NUM_BATCHED_TOKENS=6144
MAX_NUM_SEQS=6
KV_CACHE_MEMORY=8589934592
```

For a **fresh installation**, `.env.example` starts at **C4 with 8 GiB KV**.
Validate that boot and inspect memory before selecting C6.
**9 GiB remains an optional tuning profile:** set `KV_CACHE_MEMORY=9663676416`
on both nodes. We have not observed OOM locally at 9 GiB, but that does not rule
it out under heavier workloads. The active deployment now uses 8 GiB for more
RAM headroom. [External feedback](docs/EXTERNAL_FEEDBACK.md) reports a different
installation using 7.2 GiB; it does not qualify our 8 GiB profile.

Large prefills may queue under the token budget; slots do not guarantee that
all submitted requests run at once. See [profile details](docs/DEPLOYMENT.md#choose-c4-or-c6).

## Deployment

Prepare two headless Linux ARM64 GB10 hosts with Docker, Compose v2, the NVIDIA
container runtime, working RoCE and passwordless SSH from head to worker.
Follow the [full setup guide](docs/DEPLOYMENT.md#deployment) for host preparation,
pinned model downloads, device access and node settings. See
[first-boot troubleshooting](docs/DEPLOYMENT.md#troubleshooting-first-boot) for
DRM modesetting, host OOM daemons, model-cache mounts and startup failures.

On **both nodes**:

```bash
git clone https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native.git
cd glm-5.3-flash-gb10-tp2-native
docker pull technigmaai/glm-5.3-flash-gb10-tp2-native:c748079-displaykv1-arm64-cu130
cp .env.example .env
chmod 600 .env
```

Edit `.env` for each node's role/rank, addresses, fabric prefixes, SSH target and
absolute paths. Download the pinned target and draft models into each user's
default Hugging Face cache as described in the guide. `scripts/configure.py` is optional.

The image supplies the runtime; **this repository supplies the native entrypoint
and overlays through Compose bind mounts**. Use the complete recipe when starting
it. Replies have no fixed deployment output-token cap: omit `max_tokens` and
`max_completion_tokens` in client requests to use the remaining context budget.
Clients may still impose their own limits, and the model may stop earlier.

After both nodes are configured and their models are ready, run from the head:

```bash
./check.sh
./start.sh --approved
./status.sh
# Wait for API health and the startup self-test, then:
./verify.sh
```

## Operations

```bash
./check.sh
./status.sh
./tail-log.sh head --tail 80
./tail-log.sh worker --tail 80
./restart.sh --approved
./stop.sh --approved
```

The short root scripts share one implementation under `scripts/`. Each node
keeps its own private `.env`. To synchronize source from the head while both
deployment folders are offline, use `./sync-repo.sh --dry-run`, review it,
then `./sync-repo.sh --approved`. Git metadata and node settings are preserved.
Source synchronization is explicit; start does not overwrite worker files.

To switch C4/C6, edit both `.env` files and restart the pair during a planned
interruption. Changed batch sizes need matching optimized snapshots; a missing
snapshot causes a slower initial load. No automatic watchdog is installed.

See [source updates](docs/DEPLOYMENT.md#updating-an-existing-deployment) and
[layout migration and rollback](docs/LAYOUT_MIGRATION.md) for existing
installations. All runtime source belongs to this folder; models, caches and
logs keep their configured external locations.

## Validation and limits

The current **8 GiB / C6** restart passed all four model aliases, **8/8 smoke
checks** including image and tool calls, and **six simultaneous short streams**
with zero preemptions. Boot reported **1,352,535 shared KV tokens (1.29× maximum
context)**. A request with no output-token limit ended normally.
Historical 9 GiB C4/C6 checks remain recorded separately, including the larger
prefill test; no OOM was observed locally in those checks.
The reorganized layout was restarted and revalidated on **2026-10-07** with
the same results and unchanged runtime settings.

**An actual million-token workload is not qualified for this native profile.**
The KV capacity is shared; six slots do not provide six simultaneous million-token
contexts. Maximum image-count/resolution workloads are also untested. The head
has limited free RAM; the earlier 9 GiB C6 uncached load used substantial swap.

See [recorded results](manifests/validation-summary.json) and
[validation details](docs/DEPLOYMENT.md#validation-and-limits).

## Image rebuild and source maintenance

The published runtime is Linux ARM64, CUDA 13.0, PyTorch 2.13 and the pinned
Kindling/vLLM stack. Source pins, the image digest, rebuild commands and
troubleshooting are in the [full guide](docs/DEPLOYMENT.md#image-rebuild-and-source-maintenance).

## Credits and licenses

Thanks to Kindling AI for the serving stack and
[coolbho3k](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark) for the
display-reserved CUDA allocation technique, plus the model and runtime maintainers.
Copied and derived files retain their applicable licenses; see
[THIRD_PARTY.md](THIRD_PARTY.md) and the [complete credits](docs/DEPLOYMENT.md#credits-and-licenses).
