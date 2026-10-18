# Runtime releases

The public image is
`technigmaai/glm-5.3-flash-gb10-tp2-native:<version>` on Docker Hub.
All six releases are Linux ARM64 / CUDA 13.0 and use native two-node TP2,
one model container per GB10 node. Version numbers are consecutive release
identifiers, independent of patch counts. Use a fixed version tag and the
matching script in [launchers/](launchers/README.md).

| Version | Release contents |
|---|---|
| **v0.1.0** | Original native runtime, based on Kindling `c748079d45e6e070b2acb108a91edfe52f4a7747`, with display-backed KV, DFlash2, adaptive-k, RecoverSSM and weight snapshots. |
| **v0.2.0** | v0.1.0 plus the preserved [Kindling PR92](https://github.com/kindlingai/glm-5.3-flash-gx10/pull/92) ARX synchronization fix at `6f3c2b0e68ad3abebb3647335a5d5276de296e25`. |
| **v0.3.0** | v0.2.0 plus Kindling PR93, vLLM PR61213/61253/56148/59511, and FlashInfer PR6372. Adapted overlays preserve the pinned runtime's APIs. |
| **v0.4.0** | v0.3.0 plus [vLLM PR60654](https://github.com/vllm-project/vllm/pull/60654), changing two MLA prefill workspace-lifetime implementations. |
| **v0.5.0** | v0.4.0 plus Kindling PR94 and vLLM PR61174/61996: ARX, GPU block-table and sampler updates. |
| **v0.6.0** | v0.5.0 plus vLLM PR61161/61451, the equivalent final-row safety protection from PR60945, and preservation of the existing draft-truncation/acceptance-estimator guards. |

Published OCI index, ARM64 manifest and configuration digests, recipes and source layers are tracked in
[profiles/images.json](profiles/images.json). Pinned upstream PR heads and
adaptation notes are in [profiles/provenance/](profiles/provenance/). Image
identities are checked after pull on **both** nodes. Docker engines may report
the index, platform manifest or configuration digest as the local image ID;
all three are pinned. Publishing these version tags preserves the
existing images' contents and identities.

## Settings and source

Each image must be paired with its recipe: common source lives in `files/`,
and version-specific replacements in `profiles/patches/`. Compose merges bind
mounts by destination, allowing newer fixes to replace earlier overlays.
Do not run an experimental image with only the baseline Compose file.
Each version has a separate `TORCH_EXTENSIONS_DIR`. Host paths, node addresses,
model caches and serving limits remain in the private node configuration.
Launchers never depend on the author's original experiment directories.

The v0.1.0 recipe disables ARX two-shot. Patched releases use the explicit
`VLLM_ARX_TWO_SHOT_MAX_KB` setting from the node configuration, defaulting to
2048 KiB when omitted. Set it identically on both nodes. Versioned recipes
disable the automatic generation self-test; verification remains explicit.

## Validation scope

The original C4/C6, 8/9 GiB results elsewhere in the deployment documentation
refer to the original runtime and its stated settings. They do not qualify all
experimental releases. Imported patch records include offline checks; those
records do not establish GPU numerical parity or performance improvements.

The image assigned v0.6.0 was observed running on both nodes on 2026-10-18,
with 1,047,552 context, 6,144 batched tokens, 16 sequence slots, 7 GiB logical KV
per GPU and 1.75 GiB display backing within that pool. Both ranks were healthy,
had zero restarts, and the 14 sampling/trio patch hashes matched the recorded
payloads. Earlier activation evidence recorded six short sequential smoke checks.
This is not a long-context, sampling-distribution or performance qualification,
and these observed serving settings are not imposed on fresh installations.

PR60654 affects FlashInfer/TRTLLM-ragged MLA prefill paths. The observed GB10
startup uses FLASH_ATTN MLA prefill, so its reported workspace savings should
not be assumed for this deployment. The dispram experiment reuses v0.4.0 and
requires a separate host allocator; it is not included in these launchers.
