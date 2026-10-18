# Choose a runtime version

Run these scripts from the **head** after configuring this repository on both
ARM64 GB10 nodes. Each script selects an image **and its matching source overlays**.
Versions are consecutive releases, independent of how many patches they contain.

| Launcher | Image tag | Contents |
|---|---|---|
| `v0.1.0.sh` | `v0.1.0` | Original native TP2 runtime |
| `v0.2.0.sh` | `v0.2.0` | PR92 baseline |
| `v0.3.0.sh` | `v0.3.0` | Six-patch experiment |
| `v0.4.0.sh` | `v0.4.0` | v0.3.0 plus PR60654 |
| `v0.5.0.sh` | `v0.5.0` | v0.4.0 plus the trio update |
| `v0.6.0.sh` | `v0.6.0` | v0.5.0 plus sampling/speculation fixes |

Full contents, upstream pins and validation limits are in [VERSIONS.md](../VERSIONS.md).

```bash
# On BOTH nodes, in the same repository revision:
git clone https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native.git
cd glm-5.3-flash-gb10-tp2-native
# Follow docs/DEPLOYMENT.md to configure each node's own .env and cached models.
docker pull technigmaai/glm-5.3-flash-gb10-tp2-native:v0.6.0

# On the HEAD, from the repository root:
./launchers/v0.6.0.sh --dry-run
./launchers/v0.6.0.sh --approved
./status.sh

# Switch to another installed version; requests are interrupted during the switch:
./launchers/v0.2.0.sh --approved
./stop.sh --approved
```

Use `./profiles.sh` to list versions, `./status.sh --json` to inspect both ranks,
and `./check.sh --profile v0.6.0` to validate a selected version without stopping it.
`./check.sh` checks `RUNTIME_VERSION` (default `v0.1.0`); `./check.sh --all`
checks all six versions, requiring all six images on both nodes.
`./start.sh --profile v0.6.0 --approved` is equivalent to its launcher.
Without `--profile`, `start.sh` selects `RUNTIME_VERSION` from `.env`, defaulting
to `v0.1.0`. `restart.sh` and `stop.sh` select the recognized running version.
An already running version is left in place; use `restart.sh --approved` to
apply configuration or source changes.

Each node keeps a private `.env`. Optional **complete** `.env.v0.1.0` through
`.env.v0.6.0` files can provide different serving settings for individual
versions; they are never generated, copied, rewritten or committed by a launcher.
Use the same project/container names and node roles across those files. The base
`.env` supplies cluster identity and the shared lifecycle lock. `DEPLOY_ROOT`
must be this checkout on each node, and `PEER_DEPLOY_DIR` must point to the
worker's checkout. Both nodes must use the same repository revision and matching
serving parameters. Version selection overrides `IMAGE` without editing the file.

Launchers check images and configuration on both nodes before stopping either
rank, refuse unknown or mixed-version containers, and never download images or
build during a switch. After stopping, both nodes must pass the idle-resource
and fabric checks before either rank starts. Startup is asynchronous. If a step
fails, inspect both nodes with `status.sh` and `tail-log.sh`; no automatic rollback
or generation benchmark is performed.

The old external profile directory is no longer needed by this layout. A known
native deployment from an older checkout can be replaced by a versioned launcher
after configuring the new checkout on both nodes. Merely cloning, checking,
listing, publishing tags, or rendering a dry run does not restart it.

The dispram allocator variant is a separate host experiment using the v0.4.0
image; it is not a seventh image or a portable image launcher. Stop its allocator
with its own controller before switching. The upstream Mentat recipe and
TensorFold images use different deployment architectures and are outside this
native TP2 version catalog.
