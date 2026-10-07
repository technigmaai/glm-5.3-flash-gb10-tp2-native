# Display allocator provenance

The records below describe the original R28 allocator experiment. The current
native recipe adapts that helper to the pinned Kindling stack; see the
[deployment guide](../../docs/DEPLOYMENT.md#kv-capacity-and-headless-display-memory)
and [source manifest](../../manifests/source.json) for the current image and
integration. The R28 base below is historical, not a setup requirement.

- Base image: `local/vllm:glm53-karmic-r28-arm64-sm121-cu134`
- Upstream allocator: `coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark`
- Pinned upstream commit: `878e0eecd893fadc69ad2d58b2df0fabb0fae2ee`
- Upstream files: `release/runtime/sources/display_kv.c` and `release/runtime/serving/ds41/display_kv.py`
- License: AGPL-3.0-only
- Local change: raise the ordinary contiguous-prefix ceiling from 1 GiB to 16 GiB for the R28 GLM KV allocation.
- Base vLLM utils SHA-256: `260042152d342e4a9202149ebfe9d5cd3ddbf4eed1272f86d6e2faebbf67c057`
