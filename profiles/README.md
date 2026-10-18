# Runtime recipes (internal)

For normal operation, use [`launchers/launch.sh`](../launchers/README.md).
For release contents, use [`VERSIONS.md`](../VERSIONS.md).

| Path | Purpose |
|---|---|
| `images.json` | Canonical release names, image references, digest pins and short descriptions |
| `v0.x.0.compose.json` | Matching image, replacement mounts and extension cache for one release |
| `patches/` | Repository-owned source replacements, grouped by their source layer |
| `provenance/` | Pinned upstream heads, payload hashes and adaptation notes |

Shared runtime files live in `files/`. Compose merges replacement mounts by
destination, so later source layers can replace earlier patches without copying
entire deployment folders. Both nodes must use the same tracked source revision.
Source integrity is recorded in `manifests/source.json`.

The image catalog pins the OCI index, Linux ARM64 platform manifest and image
configuration digest. Docker's classic and containerd stores may report different
members of this set as the local image ID; the controller recognizes all three.

These runtime recipes are separate from C4/C6 serving settings. Private `.env`
files, model weights, generated caches and operator experiment folders do not
belong in this directory and are not required to be present in a fresh clone.
