# Source provenance and license notices

This repository derives from Kindling AI's GLM-5.3-Flash recipe at
[c748079d45e6e070b2acb108a91edfe52f4a7747](https://github.com/kindlingai/glm-5.3-flash-gx10/tree/c748079d45e6e070b2acb108a91edfe52f4a7747).
The native launcher replaces its Mentat orchestration with fixed native ranks.
Vendored `files/overlays/` files come from that pinned source; the local snapshot
selector adds exact-key lookup in a read-only seed cache.

Files with vLLM SPDX headers retain their Apache-2.0 notices. The recipe pins
vLLM at `ddd6fbca148a867aad1fcab7ec72f582b9977db4`. The Apache-2.0 text is retained
in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt), and the upstream MegaMoE
license remains in [files/overlays/megamoe/LICENSE](files/overlays/megamoe/LICENSE).
Other upstream files retain their original notices and applicable terms;
this repository does not relicense them as a single combined work.

The display allocator derives from
[coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark)
at `878e0eecd893fadc69ad2d58b2df0fabb0fae2ee`. Its AGPL-3.0-only license, C source,
compiled ARM64 helper and adaptation notes are retained under `files/display-kv/`.
See [files/display-kv/ORIGIN.md](files/display-kv/ORIGIN.md) and
[files/display-kv/LICENSE.AGPL-3.0](files/display-kv/LICENSE.AGPL-3.0).

Model weights and base container layers are not included. Their license terms
remain independent of this deployment source; see the model and image sources
linked in the README. The DFlash2 model card describes noncommercial terms and
separate commercial licensing.
