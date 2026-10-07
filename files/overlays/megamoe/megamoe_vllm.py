# SPDX-License-Identifier: Apache-2.0
"""megamoe for decode-sized batches on vLLM's NVFP4 MoE layers (GB10).

megamoe_rm.cu reads the FlashInfer CUTLASS backend's own processed tensors, so
nothing is copied: batches of at most VLLM_MEGAMOE_MAX_TOKENS tokens take
megamoe and larger ones keep CUTLASS. megamoe keeps activations in 16-bit
(W4A16, exact in the weights), where CUTLASS quantizes them to FP4.

With VLLM_MOE_PREFILL=1, batches of at least VLLM_MOE_PREFILL_MIN_TOKENS
tokens take moe_prefill.cu instead: the same W4A4 math as CUTLASS on the same
tensors, with the token gather and SwiGLU folded into the fc1 GEMM.
VLLM_MOE_PREFILL_CHECK=N also runs CUTLASS on the first N such batches and
logs the difference.

install() swaps each ModelOptNvFp4FusedMoE method's class for a subclass whose
apply() dispatches by batch size, so every isinstance check still holds. Only
layers whose finalize is synchronous qualify: there the method returns the
weighted top-k sum and the runner owns shared experts and the TP all-reduce,
which is what megamoe produces.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SRC = os.environ.get("VLLM_MEGAMOE_SRC", "/opt/megamoe/megamoe_rm.cu")
_PREFILL_SRC = os.environ.get("VLLM_MOE_PREFILL_SRC", "/opt/megamoe/moe_prefill.cu")
_PREFILL = os.environ.get("VLLM_MOE_PREFILL") == "1"
_PREFILL_MIN = int(os.environ.get("VLLM_MOE_PREFILL_MIN_TOKENS", "1024"))
_check_left = int(os.environ.get("VLLM_MOE_PREFILL_CHECK", "0"))
# fc2's per-expert rows as e4m3 with a scale per 128 columns: half the bytes
# written and read back, for callers that pass the scales on (arxbig's finalize).
_PREFILL_Y8 = os.environ.get("VLLM_MOE_PREFILL_Y8") == "1"
_ext = None
_pext = None


def _jit_load(name, *args, **kwargs):
    """torch's cpp_extension.load(), safe after a build was killed midway (#53).

    torch marks a build with a `lock` file and waits forever on one it finds, so a
    build killed midway hangs every later load with nothing in the log. Builds here
    also hold an flock, which the kernel drops when its holder dies: whoever holds
    it is the only builder, and any `lock` file it finds is stale."""
    import fcntl

    from torch.utils.cpp_extension import _get_build_directory, load

    build = _get_build_directory(name, False)
    with open(os.path.join(build, "lock.flock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            os.remove(os.path.join(build, "lock"))
        except FileNotFoundError:
            pass
        return load(name, *args, **kwargs)
_scratch: dict = {}


def _load():
    global _ext
    if _ext is None:
        _ext = _jit_load("megamoe_rm", [_SRC], extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"])
    return _ext


def _load_prefill():
    global _pext
    if _pext is None:
        _pext = _jit_load("moe_prefill", [_PREFILL_SRC], extra_cuda_cflags=[
            "-O3", "-gencode=arch=compute_121a,code=sm_121a", "-DMOE_STAGES=2", "-DMOE_MINB=2", "-DMOE_PREFETCH=5"])
    return _pext


def prefill_routed(m, x, topk_weights, topk_ids, y8: bool = False, x4=None):
    """moe_prefill.cu's fc1 and fc2 on the tensors CUTLASS would get.

    x4 = (xq, xs, block): x already quantized by scaled_fp4_quant in blocks
    of `block` rows, each padded to a multiple of 128 rows (the SP gather of
    each rank's rows). x then only gives the shape and device.

    Returns (y, pos, y8s): y [M * topk, H] holds each (token, k) expert output
    in expert-sorted order, pos [M * topk] its row. With y8, y is e4m3 bytes
    and y8s [M * topk, H / 128] fp32 its scales; otherwise y is bf16, y8s None.
    """
    fe = m._moe_prefill_experts
    w13, w2 = m.w13_weight, m.w2_weight
    E, H, I = w13.shape[0], w2.shape[1], w13.shape[1] // 2
    M, topk = topk_ids.shape
    # A row with no routed experts carries id -1 (weight 0). Every expert id is used as an
    # index below (scatter_add_, argsort/searchsorted, the fc1 gather), so point it at expert 0;
    # its weight is 0, so finalize adds nothing for it, the same as CUTLASS skipping it.
    topk_ids = topk_ids.masked_fill(topk_ids < 0, 0)
    from vllm import _custom_ops as ops

    if x4 is None:
        xq, xs = ops.scaled_fp4_quant(x, m._moe_prefill_a1)
    else:
        xq, xs, block = x4
    ids = topk_ids.to(torch.int32)
    flat = ids.flatten().long()
    R = flat.numel()
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(E, dtype=torch.int64, device=x.device).scatter_add_(0, flat, torch.ones_like(flat))
    off = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
    off[1:] = torch.cumsum(counts, 0)
    mt = (counts + 127) // 128
    tcum = torch.cumsum(mt, 0)
    tidx = torch.arange(R // 128 + E, device=x.device)
    te = torch.searchsorted(tcum, tidx, right=True)
    tm = tidx - (tcum - mt)[te.clamp(max=E - 1)]
    tiles = torch.stack([te, tm], 1).int().contiguous()
    pos = torch.empty(R, dtype=torch.int32, device=x.device)
    pos[order] = torch.arange(R, dtype=torch.int32, device=x.device)
    rows = order // topk
    if x4 is not None and block % 128:
        rows = rows // block * ((block + 127) // 128 * 128) + rows % block
    rows = rows.int()
    hq = torch.empty(R, I // 2, dtype=torch.uint8, device=x.device)
    hs = torch.empty(((R + 127) // 128) * 128 * (I // 16), dtype=torch.uint8, device=x.device)
    _pext.fc1(xq, xs.view(torch.uint8).flatten(), rows, off, tiles, w13, fe.w1_scale.view(torch.uint8),
              m._moe_prefill_g1, m._moe_prefill_a2, hq, hs, m._megamoe_limit)
    if y8:
        y = torch.empty(R, H, dtype=torch.uint8, device=x.device)
        y8s = torch.empty(R, H // 128, dtype=torch.float32, device=x.device)
        _pext.fc2(hq, hs, off, tiles, w2, fe.w2_scale.view(torch.uint8), m._moe_prefill_g2, y, y8s)
        return y, pos, y8s
    y = torch.empty(R, H, dtype=torch.bfloat16, device=x.device)
    _pext.fc2(hq, hs, off, tiles, w2, fe.w2_scale.view(torch.uint8), m._moe_prefill_g2, y)
    return y, pos, None


def _prefill(m, x, topk_weights, topk_ids):
    """moe_prefill.cu on the tensors CUTLASS would get; returns the weighted top-k sum."""
    y, pos, _ = prefill_routed(m, x, topk_weights, topk_ids)
    out = torch.empty(x.shape[0], m.w2_weight.shape[1], dtype=torch.bfloat16, device=x.device)
    _pext.finalize(y, pos, topk_weights.to(torch.float32).contiguous(), out)
    return out


def _buffers(M: int, topk: int, H: int, I: int, device) -> dict:
    """Scratch shared by every layer (they run one after another), per batch size."""
    key = (M, topk, H, I)
    if key not in _scratch:
        pairs = M * topk
        _scratch[key] = dict(
            xh=torch.empty(M, H, dtype=torch.half, device=device),
            ints=torch.empty(1 + pairs + 8 * pairs, dtype=torch.int32, device=device),
            hbuf=torch.empty(pairs, 8, I, dtype=torch.half, device=device),
            ypart=torch.empty(pairs, H, dtype=torch.float32, device=device))
    return _scratch[key]


def _apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
    global _check_left
    M = x.shape[0]
    # Rows with no routed experts arrive as id -1 / weight 0 (seen in batches that mix a long
    # prefill chunk with concurrent decode rows). CUTLASS skips them; megamoe and moe_prefill
    # index by id, so send them to expert 0 at weight 0, which adds exactly nothing.
    # Elementwise, no host sync.
    neg = topk_ids < 0
    topk_ids = topk_ids.masked_fill(neg, 0)
    topk_weights = topk_weights.masked_fill(neg, 0)
    if (_PREFILL and M >= _PREFILL_MIN and x.dtype == torch.bfloat16 and hasattr(layer, "_moe_prefill_experts")
            and not torch.cuda.is_current_stream_capturing()):
        out = _prefill(layer, x.contiguous(), topk_weights, topk_ids)
        if _check_left > 0:
            _check_left -= 1
            ref = super(type(self), self).apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
            ref = ref[0] if isinstance(ref, tuple) else ref
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            logger.warning("moe_prefill check: %d tokens, rel diff vs CUTLASS %.4f", M, rel)
        return out
    if M == 0 or M > self._megamoe_max_tokens or x.dtype != torch.bfloat16:
        return super(type(self), self).apply(layer, x, topk_weights, topk_ids, shared_experts,
                                             shared_experts_input)
    w13, w2 = layer.w13_weight, layer.w2_weight
    H, I, topk = w2.shape[1], w13.shape[1] // 2, topk_ids.shape[1]
    s = _buffers(M, topk, H, I, x.device)
    out = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _ext.forward(x.contiguous(), topk_ids.to(torch.int32).contiguous(), topk_weights.to(torch.float32).contiguous(),
                 w13, layer.w13_weight_scale, layer._megamoe_alpha1, w2, layer.w2_weight_scale,
                 layer._megamoe_alpha2, s["xh"], s["ints"], s["hbuf"], s["ypart"], out,
                 layer._megamoe_limit, self._megamoe_variant)
    return out


def install(model: torch.nn.Module) -> None:
    """Route decode-sized batches of every eligible NVFP4 MoE layer to megamoe."""
    if os.environ.get("VLLM_MEGAMOE") != "1":
        return
    max_tokens = int(os.environ.get("VLLM_MEGAMOE_MAX_TOKENS", "8"))
    variant = int(os.environ.get("VLLM_MEGAMOE_VARIANT", "20"))
    classes: dict[type, type] = {}
    count = 0
    for name, m in model.named_modules():
        method = getattr(m, "quant_method", None)
        if type(method).__name__ != "ModelOptNvFp4FusedMoE" or not hasattr(m, "w13_weight"):
            continue
        kernel = getattr(method, "moe_kernel", None)
        impl = getattr(kernel, "impl", kernel)
        pf = getattr(impl, "prepare_finalize", None)
        if method.is_monolithic or pf is None or pf.supports_async() or m.expert_map is not None:
            logger.warning("megamoe: %s does not qualify; it keeps CUTLASS", name)
            continue
        clamp = getattr(getattr(impl, "fused_experts", None), "gemm1_clamp_limit", None)
        limit = float(clamp.max()) if isinstance(clamp, torch.Tensor) else float("inf")
        m._megamoe_limit = limit
        m._megamoe_alpha1 = (m.w13_weight_scale_2 / m.w13_input_scale).float().contiguous()
        m._megamoe_alpha2 = (m.w2_weight_scale_2 / m.w2_input_scale).float().contiguous()
        fe = getattr(impl, "fused_experts", None)
        if _PREFILL and fe is not None and getattr(fe, "g1_alphas", None) is not None:
            E = m.w13_weight.shape[0]
            m._moe_prefill_experts = fe
            m._moe_prefill_a1 = fe.a1_gscale.float().flatten()[:1].contiguous()
            m._moe_prefill_g1 = fe.g1_alphas.float().flatten().expand(E).contiguous()
            m._moe_prefill_a2 = fe.a2_gscale.float().flatten().expand(E).contiguous()
            m._moe_prefill_g2 = fe.g2_alphas.float().flatten().expand(E).contiguous()
        cls = type(method)
        if cls not in classes:
            classes[cls] = type("MegaMoE" + cls.__name__, (cls,), {"apply": _apply})
        method.__class__ = classes[cls]
        method._megamoe_max_tokens = max_tokens
        method._megamoe_variant = variant
        count += 1
    if count:
        _load()
        if _PREFILL:
            _load_prefill()
    logger.info("megamoe: %d MoE layers take batches of <= %d tokens (variant %d)", count, max_tokens, variant)
