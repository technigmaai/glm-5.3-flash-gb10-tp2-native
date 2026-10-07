# Layout migration and rollback

The runtime remains Kindling-derived native TP2: one container per node, with
external model caches and the same serving settings. The root now contains
short operator scripts and one compose.yaml; runtime files live under files/,
implementation helpers under scripts/, image sources under image/ and integrity
records under manifests/. No file depends on a symlink to an older deployment.

## Prepare while the current pair serves

1. Back up each COMPLETE existing deployment folder, including hidden files,
   private .env files, ignored/untracked files and experiment evidence. Store
   archives outside the repository, restrict their access, checksum them and
   verify a test restore. Preserve Git history with a bundle plus a full source
   archive; a bundle alone excludes working-tree files.
2. Clone this same GitHub repository into a new folder on both hosts. Preserve
   its origin/history and check out the same tested layout commit.
3. Copy each node's own .env privately into its new checkout. Change DEPLOY_ROOT
   to that node's new folder and PEER_DEPLOY_DIR to the worker's new folder.
   Keep the actual installed image tag, model paths, external caches/logs,
   snapshot seed, ranks, API aliases and serving values. Do not overwrite the
   worker's .env with the head's.
4. Run ./check.sh on both nodes and render configuration with
   ./scripts/cluster.sh config. Compare the new configuration with the old:
   serving values, devices, ports, image contents, model/cache/log mounts and
   runtime overlay bytes should match. Source paths and the relocated
   entrypoint/healthcheck locations are intentional differences.
5. Keep the new checkout staged until cutover. With the same names/ports/GPU
   resources it cannot serve alongside the old pair. Idle-resource preflight
   is for stopped ranks; do not run it against an intentionally live model.

The CPU-only regression suite is `python3 -m unittest discover -s tests -v`.
`sync-repo.sh --dry-run` and `sync-repo.sh --approved` copy tracked source only,
never .env or Git metadata, and refuse folders mounted by running containers.
Prefer matching Git checkouts when deploying a published revision. Source sync
has no --delete; commit/check out the same source revision separately on each
node so Git history agrees with the files.

## Planned cutover

1. Stop both ranks using the OLD deployment's stop command.
2. From the NEW head checkout, run ./start.sh --approved. The launcher checks
   both nodes, then follows the existing native rendezvous sequence.
3. Wait for head health, exact-key snapshot restoration and the startup self-test.
4. Run ./status.sh and ./verify.sh. Verification submits finite alias, tool,
   image and short concurrent requests; it does not qualify million-token use.
5. Confirm all deployment bind-source paths point to the new folder.

Existing complete compatible snapshots retain their keys. No cold conversion,
image pull or rebuild is required merely because the source folder moved.
No automatic watchdog or host-service change is introduced by this layout.

## Rollback

Stop the new pair with ./stop.sh --approved, then start the preserved OLD
folder with its original per-node .env, image and caches. Keep that folder and
verified archives until the migration has been accepted. If source files need
restoration, extract the per-node archive into a separate location first and
verify it before replacing anything.

If a published Git change needs reversal, use a normal git revert. Preserve
history; do not force-push or publish private backups. Never delete model
weights, caches, images or evidence as part of a source-layout rollback.

## Command and path changes

| Previous | New |
|---|---|
| ./cluster.sh start/stop/restart --approved | ./start.sh, ./stop.sh, ./restart.sh --approved |
| ./cluster.sh status | ./status.sh |
| ./cluster.sh logs | ./tail-log.sh head or worker |
| ./cluster.sh check / verify | ./check.sh / ./verify.sh |
| compose.json | compose.yaml |
| entrypoint.sh, healthcheck.py | files/entrypoint.sh, files/healthcheck.py |
| experimental/ | files/overlays/ |
| display-kv/ | files/display-kv/; Dockerfile under image/display-kv/ |
| manifest.json, copy-assets.json, validation-summary.json | manifests/source.json, manifests/copy-assets.json, manifests/validation-summary.json |
| configure.py, install-assets.py | scripts/configure.py, scripts/install-assets.py (optional) |

The optional previous Mentat-stack adapter remains under scripts/legacy.sh;
it is not needed for normal native serving or for rollback to the preserved
native folder. Advanced internal commands remain in scripts/cluster.sh.
