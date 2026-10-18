# Image versions

Image: `technigmaai/glm-5.3-flash-gb10-tp2-native:<version>`

All six images are published on [Docker Hub](https://hub.docker.com/r/technigmaai/glm-5.3-flash-gb10-tp2-native/tags)
for **Linux ARM64 / NVIDIA GB10**. Versions are consecutive release identifiers,
independent of patch counts. Each release builds on the preceding one.

Select the image from the menu:

```bash
./launchers/launch.sh
```

Or use `./launchers/launch.sh start --version v0.6.0 --approved` after preparing
both nodes. The launcher selects the matching repository source overlays as well
as the image. See [installation](docs/DEPLOYMENT.md) before starting a fresh pair.

| Version | Main change | Status |
|---|---|---|
| [v0.1.0](#v010) | Original native TP2 recipe | Original baseline |
| [v0.2.0](#v020) | ARX synchronization | Patched baseline |
| [v0.3.0](#v030) | Six backports across collectives, parser, attention and drafting | Experiment |
| [v0.4.0](#v040) | MLA prefill workspace lifetime | Experiment |
| [v0.5.0](#v050) | Relay handling, block-table cleanup and sampler | Experiment |
| [v0.6.0](#v060) | Speculative randomness and verification | Experiment; operator deployment observed 2026-10-18 |

These labels do not rank performance or establish production readiness. Existing
long-form tags remain compatibility references; the public launcher uses fixed
version tags and verifies their contents against the pinned digest catalog.

## Shared foundation

All releases retain native vLLM two-node TP2, one container per node, an
OpenAI-compatible head API, DFlash2, adaptive-k, RecoverSSM, weight snapshots and
display-backed KV. Model weights stay outside the images.

| Component | Pin |
|---|---|
| Kindling base | `c748079d45e6e070b2acb108a91edfe52f4a7747` |
| vLLM base | `ddd6fbca148a867aad1fcab7ec72f582b9977db4` (`0.30.1rc1.dev193`) |
| CUDA / PyTorch | 13.0 / 2.13.0+cu130 |
| FlashInfer / Triton | 0.7.0 / 3.7.1 |

The images contain scoped backports on that foundation, not wholesale upgrades
to the current upstream branches. PR links describe intent; the imported commit
pins and adaptation notes identify the exact included source snapshot.

## v0.1.0

**Original native runtime.** Based on the pinned Kindling source, with the native
rank launcher and display allocator. ARX two-shot is forced off by this recipe.
The original C4/C6 validation belongs to this baseline and its recorded settings.

## v0.2.0

**ARX synchronization baseline.** Adds [Kindling PR92](https://github.com/kindlingai/glm-5.3-flash-gx10/pull/92)
at `6f3c2b0e68ad3abebb3647335a5d5276de296e25`: the two-shot scatter waits for this
rank's own publish. Its source replacements are retained under `profiles/patches/pr92/`.

## v0.3.0

**Six-patch experiment.** Adds these adapted backports to v0.2.0:

| Source | Intended change |
|---|---|
| [Kindling PR93](https://github.com/kindlingai/glm-5.3-flash-gx10/pull/93) | Use 64-bit GPU collective counters. |
| [vLLM PR61213](https://github.com/vllm-project/vllm/pull/61213) | Improve top-k/top-p handling of masked logits. |
| [FlashInfer PR6372](https://github.com/flashinfer-ai/flashinfer/pull/6372) | Fix KDA prefill metadata caching with inference tensors. |
| [vLLM PR61253](https://github.com/vllm-project/vllm/pull/61253) | Stream GLM tool arguments without repeatedly parsing the whole argument. |
| [vLLM PR56148](https://github.com/vllm-project/vllm/pull/56148) | Adapt multi-query Triton verification and segment/padding handling. |
| [vLLM PR59511](https://github.com/vllm-project/vllm/pull/59511) | Capture draft context-combine and anchor work in the CUDA graph. |

Included snapshots may precede the PR's current head. See [six-patch pins and
adaptations](profiles/provenance/sixpatch.json) for exact revisions and payload hashes.

## v0.4.0

**MLA prefill workspace-lifetime experiment.** Adds [vLLM PR60654](https://github.com/vllm-project/vllm/pull/60654)
at `df9307fa185655a6e779b1aff3e9e033841ade4e`, modifying FlashInfer and TRTLLM-ragged
MLA prefill to obtain workspaces at use time rather than retaining them from initialization.

The observed GB10 startup uses **FLASH_ATTN MLA prefill**, so workspace savings
reported for the other backends must not be assumed for this deployment. The
separate host dispram experiment reuses this image; it is not a seventh image
and is not installed by the portable launcher.

## v0.5.0

**Relay, block-table and sampler experiment.** Adds to v0.4.0:

- [Kindling PR94](https://github.com/kindlingai/glm-5.3-flash-gx10/pull/94): report ARX/arxbig relay failure explicitly.
- [vLLM PR61174](https://github.com/vllm-project/vllm/pull/61174): clear stale shared GPU block-table rows.
- [vLLM PR61996](https://github.com/vllm-project/vllm/pull/61996): change the small-batch split top-p pipeline.

Exact pins and payload hashes: [trio source record](profiles/provenance/trio.json).

## v0.6.0

**Speculative sampling and verification experiment.** Adds to v0.5.0:

- [vLLM PR61161](https://github.com/vllm-project/vllm/pull/61161): avoid reusing random draws across verification rounds.
- [vLLM PR61451](https://github.com/vllm-project/vllm/pull/61451): avoid unnecessary bonus-logit work for short drafts.
- Equivalent bounded-lookahead protection from [vLLM PR60945](https://github.com/vllm-project/vllm/pull/60945), while retaining existing local placeholder and rejection-handling guards. This is not a claim that the entire third PR was applied separately.

Existing draft-truncation and acceptance-estimator guards are preserved. Exact
pins, port notes and payload hashes: [sampling source record](profiles/provenance/sampling.json).

This is the exact image observed running on the operator's two nodes on
2026-10-18. Publishing the version tag did not rebuild or restart it.

## Validation

Validation must be read with the image, settings, workload and date it describes.

| Evidence | What it establishes |
|---|---|
| CPU regression tests and actual Compose rendering | Launcher logic, source packaging and recipe configuration; not GPU numerical quality. |
| Read-only image/model/source checks on both nodes and switch plans | Recipe and cache readiness without interrupting the model pair. |
| Original C4/C6 results | The original runtime's stated smoke, concurrency and tool/reasoning results. |
| v0.6.0 deployment observation | Both ranks healthy with zero restarts; 14 sampling/trio patch hashes matched recorded payloads. |
| Earlier v0.6.0 activation record | Six short sequential smoke checks; not a sampling-distribution, long-context or performance qualification. |

The observed v0.6.0 deployment used 1,047,552 context, 6,144 batched tokens,
16 sequence slots and 7 GiB logical KV per GPU, including 1.75 GiB display backing.
Those are operator settings, **not fresh-install defaults**. Fresh setup defaults
to C4 / 8 GiB. Capacity is shared, and arbitrary million-token accuracy or
maximum image-count/resolution workloads are not universally qualified.

Original evidence: [validation summary](manifests/validation-summary.json),
[dated reference measurements](docs/REFERENCE.md#validation-and-limits),
[tool/reasoning checks](docs/TOOL_REASONING_VALIDATION.md) and
[external feedback](docs/EXTERNAL_FEEDBACK.md). Results do not automatically
transfer to later experimental images.

## Digests and source

[profiles/images.json](profiles/images.json) records the public OCI index,
Linux ARM64 manifest and configuration digests. Docker image stores can report
different members of that set as the local image ID; the controller accepts
only pinned identities. [profiles/](profiles/README.md) explains the overlay layout.
Each version uses a separate compiled-extension cache. Private node settings
remain separate from source and are not rewritten when switching releases.
