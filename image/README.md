# Runtime image sources

This layout change reuses the published ARM64 image; it does not rebuild it.
The native entrypoint and pinned overlays are supplied through repository bind
mounts. See [source provenance and licenses](../THIRD_PARTY.md), the
[deployment guide](../docs/DEPLOYMENT.md#image-rebuild-and-source-maintenance)
and [source pins](../manifests/source.json).

To reconstruct the pinned base on a compatible ARM64 GB10 build host:

```bash
git clone https://github.com/kindlingai/glm-5.3-flash-gx10.git kindling-source
cd kindling-source
git checkout c748079d45e6e070b2acb108a91edfe52f4a7747
git submodule update --init --recursive
TAG=local/glm53-native:gb10-c748079-base image/build.sh
```

From the native deployment root, build the display layer using its source and
compiled helper as the context:

```bash
docker build -f image/display-kv/Dockerfile \
  --build-arg BASE_IMAGE=local/glm53-native:gb10-c748079-base \
  -t technigmaai/glm-5.3-flash-gb10-tp2-native:c748079-displaykv1-arm64-cu130 \
  files/display-kv/
```

Validate a rebuild away from a live serving workload. Original licenses and
allocator C/Python source remain beside the helper under files/display-kv/.
