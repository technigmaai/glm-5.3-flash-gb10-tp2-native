# Guided setup validation — 2026-10-10

The initial PR #5 revision (`3fd633b`) was tested in separate temporary source folders on two ARM64 GB10
nodes while their existing model deployment remained running.

| Check | Result |
|---|---|
| Repository regression suite | 78 tests passed on Linux ARM64/Python 3.12 |
| Default automatic discovery | Identified both RDMA fabric links and the actual SSH endpoints |
| Explicit management-address overrides | Selected the intended LAN pair independently of SSH's fabric route |
| Temporary network probes | Both directions passed on LAN and both fabric links |
| Configuration generation | Created separate head/worker `.env` files, mode 600 |
| Repeat setup | Refused to overwrite existing `.env` files |
| Full asset-preparation flow | Public image pulled on both nodes; both pinned models reused from their existing caches |
| Model paths | Used each user's original Hugging Face cache and the pinned revisions |
| Deployment checks | Head and worker configuration, model symlinks, mounts, image and source integrity passed |
| Existing deployment preservation | Both `.env` hashes and container start times unchanged; zero restarts and healthy API |

The first configuration check used the installed local image because the public
tag was initially absent. The complete fresh setup was then exercised from two
new Git-linked checkouts with the actual public image and real Hugging Face CLI
1.33.0. Registry pulls resolved the documented ARM64 digest. Both pinned models
were complete (44 target files and five draft files per node); dry runs reported
zero download bytes, and real `hf download` commands reused those cache snapshots.
No model weights were copied or relocated. CLI dependencies were installed only
in temporary test directories and supplied through a test-only SSH environment;
system Python and permanent SSH configuration were not modified.

This verifies registry access and cache reuse, not an empty-cache transfer of the
entire checkpoint. A new installation still needs documented tools, model access,
free disk space and network access. Preparation commands were also tested with
fake executables to check quoting and command selection without downloads.

The unit tests include adjacent `/30` and `/31` fabrics, ambiguous/missing links,
MTU/RDMA prerequisites, SSH argument preservation, exact-peer inbound/outbound
UFW rules, private plan permissions, root-only DRM diagnostics, HF cache-variable
precedence, user-local CLI discovery, missing-tool rejection, model ID/revision validation, Docker live-mount protection, exclusive
configuration creation and temporary TCP listener cleanup/failure handling.

Firewall scripts were reviewed/generated, not applied on the live nodes. TCP
probes are reachability diagnostics, not RDMA bandwidth or GPU collective tests.
This validation covers setup and configuration; it does not claim a new model
startup or inference benchmark. Runtime entrypoint, Compose service architecture
and serving profile were not changed by this feature.

## Direct HF-ID loading follow-up

The follow-up changes the loading argument to the target and draft HF IDs with
separate pinned commit revisions. Fresh configuration now records one HF hub
cache location instead of model snapshot paths. The launcher selects either the
read-only HF-cache mount or the explicit-path compatibility mounts.

Before cutover, 86 CPU regression tests passed on each node. Both the HF-ID and
existing explicit-path configurations passed real Docker Compose/model checks.
Temporary CPU-only containers with networking disabled resolved both pinned IDs,
loaded the target tokenizer, and matched the installed processed-snapshot
fingerprints for the target and draft on each node. A dry run with the complete
Compose environment and mounts checked the actual target ID/`--revision` and
draft ID/speculative revision arguments without starting another inference engine.
These checks do not constitute a long-context inference benchmark; live smoke
results are recorded separately in the PR's validation notes.
