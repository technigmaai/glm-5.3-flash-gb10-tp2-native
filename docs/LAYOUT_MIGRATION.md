# Migrate or update an existing deployment

The public entry point is now **`./launchers/launch.sh`**. The former root
operator scripts and six per-version scripts are consolidated into one terminal
menu with equivalent command-line actions. Runtime source remains in `files/`,
versioned recipes in `profiles/`, and internal helpers in `scripts/`.

## Existing deployment: prepare a separate checkout

1. Preserve each node's existing complete deployment folder and private settings.
2. Clone this repository into a new directory on both hosts and use the same commit.
3. Copy each node's own `.env` privately into its new checkout. Update `DEPLOY_ROOT`
   for that node and `PEER_DEPLOY_DIR` for the worker's new checkout. Preserve model,
   cache, log, snapshot, address and serving settings. Never copy head settings onto the worker.
4. Pull the selected version tag on both nodes and run a read-only check and plan
   from the new head checkout:

   ```bash
   ./launchers/launch.sh check --version v0.6.0
   ./launchers/launch.sh start --version v0.6.0 --dry-run
   ```

5. During the chosen interruption, apply the switch:

   ```bash
   ./launchers/launch.sh start --version v0.6.0 --approved
   ./launchers/launch.sh status
   ```

The controller recognizes catalog image identities in an older native Compose
project and recreates the pair using the selected repository recipe. It refuses
unknown images/projects and mixed-version pairs. Cloning, checking and dry-run
planning leave the running model pair in place. Old profile symlinks and
machine-specific experiment folders are not dependencies of the new checkout.

The dispram variant uses a separate host allocator and the v0.4.0 image. Stop
that variant with its own controller before changing deployment architectures.
It is not an additional portable image release. Mentat and TensorFold deployments
also require their own lifecycle handling.

## Update a checkout already used by a running pair

Stop the pair before changing bind-mounted source. From the head:

```bash
./launchers/launch.sh stop --approved
```

Then update both Git checkouts to the same revision, preserving each node's
private settings, and Check/Start the chosen version again. Do not reset or
overwrite local configuration or uncommitted source to force an update.

The optional `sync --dry-run` and `sync --approved` operations copy tracked
source while both source folders are offline. They preserve private settings
and Git metadata and refuse source mounted by running containers. Matching Git
checkouts are the clearest way to deploy a published revision.

## Command migration

| Former entry point | Single launcher equivalent |
|---|---|
| `setup.sh` | `launchers/launch.sh setup` |
| `launchers/v0.6.0.sh --approved` | `launchers/launch.sh start --version v0.6.0 --approved` |
| `start.sh --profile v0.6.0` | `launchers/launch.sh start --version v0.6.0` |
| `status.sh`, `stop.sh`, `restart.sh` | `launchers/launch.sh status`, `stop`, `restart` |
| `check.sh --profile v0.6.0` | `launchers/launch.sh check --version v0.6.0` |
| `tail-log.sh worker --tail 80` | `launchers/launch.sh logs --node worker --tail 80` |
| `verify.sh`, `sync-repo.sh` | `launchers/launch.sh verify`, `sync` |

Commands that change containers still accept `--approved`; Start also supports
`--dry-run`. The menu presents the same choices interactively. For fresh
installations, use [the deployment guide](DEPLOYMENT.md).
