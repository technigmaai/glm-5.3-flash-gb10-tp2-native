# GLM-5.3-Flash on two NVIDIA GB10 nodes

**Technigma AI** · DECODE. AUTOMATE. EVOLVE.

Run GLM-5.3-Flash NVFP4 on two Linux ARM64 GB10 systems, such as NVIDIA DGX Spark
or ASUS GX10. Native vLLM TP2 uses **one model container per node** and serves an
OpenAI-compatible API from the head on port **8000**.

[Installation](docs/DEPLOYMENT.md) · [Image versions](VERSIONS.md) ·
[Docker Hub](https://hub.docker.com/r/technigmaai/glm-5.3-flash-gb10-tp2-native) ·
[Documentation](docs/README.md)

## One launcher

```bash
./launchers/launch.sh
```

Use **Up/Down** and **Enter** to choose an operation and image version. The menu
includes setup, start/switch, status, restart, stop, checks and logs. It runs in
your terminal over SSH; no browser or graphical desktop is required.

The same file supports explicit commands:

```bash
./launchers/launch.sh list
./launchers/launch.sh start --version v0.1.0 --dry-run
./launchers/launch.sh start --version v0.1.0 --approved
./launchers/launch.sh status
```

Configure both nodes before starting. Follow the [installation guide](docs/DEPLOYMENT.md)
for host preparation, model downloads and the complete first-run sequence.

## Choose an image

All tags below use `technigmaai/glm-5.3-flash-gb10-tp2-native:<version>`.
Version numbers are consecutive release identifiers, independent of patch counts.

| Version | What changes | Role |
|---|---|---|
| [v0.1.0](VERSIONS.md#v010) | Original native TP2 runtime | Original baseline |
| [v0.2.0](VERSIONS.md#v020) | ARX synchronization fix | Patched baseline |
| [v0.3.0](VERSIONS.md#v030) | Collective counters, tool streaming, attention and draft-graph changes | Experiment |
| [v0.4.0](VERSIONS.md#v040) | MLA prefill workspace lifetime | Experiment |
| [v0.5.0](VERSIONS.md#v050) | Relay failure handling, block-table cleanup and sampler changes | Experiment |
| [v0.6.0](VERSIONS.md#v060) | Speculative sampling randomness and verification fixes | Experiment; operator's running image observed 2026-10-18 |

Use v0.1.0 to reproduce the documented original runtime. v0.6.0 identifies the
operator's observed running image; it is not a claim of production qualification
or a performance ranking. Other releases remain available for comparison and rollback.
Detailed changes, upstream pins and validation limits are in [VERSIONS.md](VERSIONS.md).
Historical long-form Docker tags remain compatibility references; use the version tags above.

## Requirements and behavior

- Two headless Linux ARM64 GB10 nodes, with Docker Engine, Compose v2 and NVIDIA Container Toolkit.
- Working RoCE links, DRM access for display-backed KV, and passwordless SSH from head to worker.
- The same repository revision and a separate private `.env` on each node.
- The pinned target and DFlash2 models cached on both nodes; weights are not included in the images.

Each version selects **both an image and its matching source overlays**. The
repository supplies the native entrypoint and bind-mounted patches, so pulling
an image alone is insufficient. Native ranks replace Mentat/Ray services.

Image version and serving settings are separate choices. Fresh setup defaults to
C4: 4,096 batched tokens, four sequence slots and 8 GiB logical KV per GPU. The
configured context is 1,047,552 tokens. These are configuration limits, not a
promise about arbitrary million-token accuracy or concurrent long-context capacity.
See [serving settings](docs/DEPLOYMENT.md#serving-settings).

## Repository map

| Location | Purpose |
|---|---|
| [`launchers/`](launchers/README.md) | The single public operator entry point, `launch.sh` |
| [`VERSIONS.md`](VERSIONS.md) | Human-readable release contents and validation scope |
| [`docs/`](docs/README.md) | Installation, migration, reference material and Docker Hub overview source |
| [`profiles/`](profiles/README.md) | Pinned images, Compose recipes and version-specific source patches |
| `files/` | Shared runtime entrypoint, chat template and overlays |
| `scripts/` | Internal implementation of the launcher and deployment helpers |
| `image/`, `manifests/`, `tests/` | Image rebuild notes, source integrity records and verification |

Models, caches, logs, private settings and local experiment records stay outside
tracked source. Launchers check both nodes before changing containers, share a
lifecycle lock, and require explicit approval for command-line switches.

## Validation and credits

The unified launcher is checked with CPU tests, Compose rendering, two-node
configuration checks and non-mutating switch plans. Version-specific inference
results and their limits are documented separately in [VERSIONS.md](VERSIONS.md).
Automatic generation self-tests are disabled in versioned recipes; inference
verification is an explicit operation.

Derived from [Kindling AI's GLM recipe](https://github.com/kindlingai/glm-5.3-flash-gx10)
and the [coolbho3k display-memory technique](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark).
Source notices remain in [THIRD_PARTY.md](THIRD_PARTY.md). Review the target and
draft model licenses separately; image publication does not change their terms.
