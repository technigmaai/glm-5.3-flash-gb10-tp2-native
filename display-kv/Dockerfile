ARG BASE_IMAGE=spark-glm53:gx10-c748079-mentat0171
FROM ${BASE_IMAGE}
COPY glm53_display_kv.py /usr/local/lib/python3.12/dist-packages/glm53_display_kv.py
COPY worker_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py
COPY libglm53_display_kv.so LICENSE.AGPL-3.0 ORIGIN.md display_kv_r28.c /opt/glm53-display-kv/
RUN python3 -m py_compile /usr/local/lib/python3.12/dist-packages/glm53_display_kv.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py \
    && echo 'ad142abfcf25a9ea91ca03acc2729decefd2a5cddf8c7d0316a48d3d2d37cde5  /opt/glm53-display-kv/libglm53_display_kv.so' | sha256sum -c -
LABEL local-inference.display-kv.bytes="1879048192" \
      local-inference.base.commit="c748079d45e6e070b2acb108a91edfe52f4a7747"
