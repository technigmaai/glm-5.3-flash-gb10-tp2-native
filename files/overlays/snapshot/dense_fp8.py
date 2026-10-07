# SPDX-License-Identifier: Apache-2.0
"""FP8 weights for the dense linears a checkpoint left in bf16.

GLM-5.3-Flash's NVFP4 checkpoint quantizes only the routed experts. About 4 GB
per rank of dense projections (the KDA input projection, o_proj, the shared
experts, the MLA and indexer query projections) stay bf16, and decode reads
all of it every step. On GB10 those GEMMs run at the memory roofline, so the
bytes are the cost: FP8 halves them.

Weights get one scale per output channel, activations one per token, and the
GEMM is CUTLASS's scaled_mm, which reaches ~90% of the FP8 read roofline from
M = 1 upwards. The router gate stays bf16 (routing is sensitive), and so do
layers whose weight other code reads directly (kv_b_proj, the indexer's
wk_weights_proj). The lm_head converts too unless VLLM_DENSE_FP8_LM_HEAD=0.

Layers whose names match VLLM_DENSE_W4 go to NVFP4 instead (W4A16 through
megadense4.cu, for up to 64 tokens where _W4_PLAN says so and 32 elsewhere;
larger batches use an FP8 copy kept alongside). Half the bytes again, at a
quality cost that depends on the layer: the KDA in_proj alone cost about as much NLL as FP8 on
everything, all dense layers about 2-3x that. The DFlash drafter's layers
only change acceptance, never the output.
"""

import os
import re

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)

_ROWS = 2048
# megadense4 launch per (N, K) weight shape and 8-row bucket of the batch (1-8,
# 9-16, ..., 57-64): the variant (tens digit K split, units digit tiles per
# block, 1 = 4 tiles unsplit), or None for the FP8 copy: the fastest of every
# variant and CUTLASS FP8 on GB10 for each TP=4 shape. Past 24 rows every block
# re-reads the whole batch, so fewer warps per tile and fewer blocks win. Other
# shapes use megadense4's default launch up to 32 rows, and FP8 above.
_W4_PLAN = {
    (6416, 4096): [81, 81, 41, 82, 41, 41, 41, 1],               # KDA in_proj
    (4096, 4096): [81, 81, 81, 82, 21, 21, 21, 21],              # MLA o_proj
    (4096, 2048): [42, 42, 21, 21, 21, 21, 21, 21],              # KDA o_proj
    (1024, 4096): [82, 82, 82, None, None, None, None, None],    # shared gate_up
    (4096, 512): [41, 21, 21, 21, 21, 21, None, None],           # shared down
}
_L2_WEIGHT_BYTES = 16 << 20
_EXCLUDE = re.compile(r"(^|\.)(gate|kv_b_proj|wk_weights_proj|index_kpool_compress_gate)$|(^|\.)visual\.")


# Activations already quantized to per-token FP8 elsewhere (the SP gather),
# keyed by the placeholder tensor handed to the layer in their place.
_prequant: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def prequant(placeholder: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor) -> None:
    """The next FP8 linear called on `placeholder` uses (xq, xs) as its input."""
    _prequant[placeholder.data_ptr()] = (placeholder, xq, xs)


def _fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                bias: torch.Tensor | None) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.shape[0] == 0:
        return x.new_empty(*shape[:-1], weight.shape[0])
    pq = _prequant.pop(x.data_ptr(), None) if _prequant else None
    if pq is not None and pq[0] is x:
        xq, xs = pq[1], pq[2]
    else:
        xq, xs = ops.scaled_fp8_quant(x2.contiguous(), use_per_token_if_dynamic=True)
    M = xq.shape[0]
    if M <= _ROWS or weight.numel() <= _L2_WEIGHT_BYTES or weight.shape[0] % 16:
        out = ops.cutlass_scaled_mm(xq, weight.t(), xs, scale, x.dtype, bias)
        return out.reshape(*shape[:-1], weight.shape[0])
    # CUTLASS rasterizes along N, so every row of tiles rereads the whole
    # weight. Once it outgrows L2 that comes from DRAM: 16k x 6416 x 4096 runs
    # at 68 TFLOPS in one call and 169 in 2048-row calls.
    out = torch.empty(M, weight.shape[0], dtype=x.dtype, device=x.device)
    for m0 in range(0, M, _ROWS):
        torch.ops._C.cutlass_scaled_mm(out[m0:m0 + _ROWS], xq[m0:m0 + _ROWS], weight.t(),
                                       xs[m0:m0 + _ROWS], scale, bias)
    return out.reshape(*shape[:-1], weight.shape[0])


class Fp8DenseLinearMethod(LinearMethodBase):
    """A converted layer's forward: per-token FP8 activations x FP8 weight.

    Not an UnquantizedLinearMethod: code that sees one assumes a bf16 weight it
    can use directly (the DFlash drafter concatenates its KV weights)."""

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError("layers are converted after loading")

    def apply(self, layer, x, bias=None):
        return _trim(layer, _fp8_linear(x, layer.weight, layer.weight_scale, bias))


def _trim(layer, y: torch.Tensor) -> torch.Tensor:
    """Drop the zero rows _pad_rows16 added (TP=3 KDA in_proj: N=8726)."""
    n = getattr(layer, "n_trim", None)
    return y if n is None else y[..., :n]


def _pad_rows16(module: torch.nn.Module) -> None:
    """Zero-extend the output rows to a multiple of 16 so the layer can
    convert (megadense4 and the tiled layout need N % 16 == 0). Outputs are
    trimmed back in apply."""
    w = module.weight.data
    N = w.shape[0]
    pad = -N % 16
    if pad == 0:
        return
    module.weight = Parameter(torch.cat([w, w.new_zeros(pad, w.shape[1])]), requires_grad=False)
    module.n_trim = N


class Fp8LMHeadMethod:
    """The lm_head's logits (the only path a ParallelLMHead takes at decode)."""

    def __init__(self, inner):
        self.inner = inner

    def apply(self, layer, x, bias=None):
        return _fp8_linear(x, layer.weight, layer.weight_scale, bias)

    def __getattr__(self, name):
        return getattr(self.inner, name)


_W4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_w4_ext = None


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


def _w4():
    global _w4_ext
    if _w4_ext is None:
        _w4_ext = _jit_load("megadense4", [os.environ.get("VLLM_MEGADENSE4_SRC", "/opt/megamoe/megadense4.cu")],
                            extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"])
    return _w4_ext


class W4DenseLinearMethod(LinearMethodBase):
    """A layer converted to NVFP4 in megadense4.cu's tiled layout."""


    def create_weights(self, *args, **kwargs):
        raise NotImplementedError("layers are converted after loading")

    def apply(self, layer, x, bias=None):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        N, K = layer.w4_shape
        if x2.shape[0] == 0:
            return _trim(layer, x.new_empty(*shape[:-1], N))
        M = x2.shape[0]
        plan = _W4_PLAN.get((N, K))
        variant = (plan[(M - 1) // 8] if M <= 64 else None) if plan else (0 if M <= 32 else None)
        if variant is None or x2.dtype != torch.bfloat16:
            return _trim(layer, _fp8_linear(x, layer.weight_fp8, layer.weight_fp8_scale, bias))
        y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        _w4_ext.gemm(x2.contiguous(), layer.weight, layer.weight_scale, layer.weight_scale_2, y, variant)
        if bias is not None:
            y += bias
        return _trim(layer, y.reshape(*shape[:-1], N))


class W4LMHeadMethod(W4DenseLinearMethod):
    """The lm_head in NVFP4 (selected by VLLM_DENSE_W4 matching "lm_head")."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _quantize_w4(module: torch.nn.Module) -> int:
    """NVFP4 with round to nearest: e4m3 scale per 16 k, one fp32 scale per
    tensor, packed into megadense4.cu's tiled layout. Also keeps an FP8 copy
    for the batch sizes where megadense4 loses to CUTLASS (see _W4_PLAN)."""
    w = module.weight.data
    N, K = w.shape
    wq8, ws8 = ops.scaled_fp8_quant(w.contiguous(), use_per_token_if_dynamic=True)
    module.weight_fp8 = Parameter(wq8, requires_grad=False)
    module.weight_fp8_scale = Parameter(ws8.view(1, -1).to(torch.float32).contiguous(), requires_grad=False)
    wf = w.float().view(N, K // 16, 16)
    gscale = (wf.abs().amax().clamp(min=1e-12) / (6.0 * 448.0)).item()
    bs = (wf.abs().amax(-1, keepdim=True) / 6.0 / gscale).to(torch.float8_e4m3fn).float()
    bs = torch.where(bs == 0, torch.ones_like(bs), bs)
    v = wf / (bs * gscale)
    grid = torch.tensor(_W4_GRID, device=w.device)
    idx = torch.bucketize(v.abs().clamp(max=6.0), (grid[1:] + grid[:-1]) / 2)
    codes = (idx + 8 * (v < 0)).to(torch.uint8).view(N, K)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    wt = packed.view(N // 16, 2, 8, K // 128, 4, 16).permute(0, 3, 1, 2, 4, 5).contiguous().view(N, K // 2)
    st = (bs.to(torch.float8_e4m3fn).view(torch.uint8).view(N, K // 16).view(N // 16, 2, 8, K // 128, 4, 2)
          .permute(0, 3, 2, 4, 1, 5).contiguous().view(-1).view(torch.int32))
    module.weight = Parameter(wt, requires_grad=False)
    module.weight_scale = Parameter(st, requires_grad=False)
    # A tensor, so weight snapshots save and restore it with the weights.
    module.weight_scale_2 = Parameter(torch.tensor([gscale], dtype=torch.float32, device=w.device),
                                      requires_grad=False)
    module.w4_shape = (N, K)
    return w.numel() * w.element_size() - wt.numel() - st.numel() * 4  # bytes a decode step no longer reads


def _quantize(module: torch.nn.Module) -> int:
    w = module.weight.data
    wq, ws = ops.scaled_fp8_quant(w.contiguous(), use_per_token_if_dynamic=True)
    module.weight = Parameter(wq, requires_grad=False)
    module.weight_scale = Parameter(ws.view(1, -1).to(torch.float32).contiguous(), requires_grad=False)
    return w.numel() * w.element_size() - wq.numel()


@torch.no_grad()
def convert(model: torch.nn.Module) -> None:
    """Convert every eligible bf16 linear in place. Idempotent."""
    if os.environ.get("VLLM_DENSE_FP8") != "1":
        return
    lm_head = os.environ.get("VLLM_DENSE_FP8_LM_HEAD", "1") == "1"
    w4 = re.compile(os.environ["VLLM_DENSE_W4"]) if os.environ.get("VLLM_DENSE_W4") else None
    saved, count, count4 = 0, 0, 0
    # Hand freed bf16 blocks back as we go: at TP=2 a rank holds ~91 GiB of
    # weights on a 121.6 GiB box, and the allocator would otherwise keep every
    # replaced weight reserved until the end and get the worker OOM-killed.
    torch.cuda.empty_cache()
    freed = 0
    for name, m in model.named_modules():
        if freed >= 1 << 30:
            torch.cuda.empty_cache()
            freed = 0
        w = getattr(m, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or w.dim() != 2:
            continue
        if w.shape[1] % 16 or _EXCLUDE.search(name):
            continue
        method = getattr(m, "quant_method", None)
        if w.shape[0] % 16:
            if not (isinstance(m, LinearBase) and isinstance(method, UnquantizedLinearMethod)
                    and getattr(m, "bias", None) is None):
                continue
            _pad_rows16(m)
            w = m.weight
        if (w4 is not None and w4.search(name) and isinstance(m, LinearBase)
                and isinstance(method, UnquantizedLinearMethod) and w.shape[1] % 128 == 0):
            _w4()
            saved += _quantize_w4(m)
            freed += w.numel() * w.element_size()
            m.quant_method = W4DenseLinearMethod()
            count4 += 1
        elif isinstance(m, LinearBase) and isinstance(method, UnquantizedLinearMethod):
            saved += _quantize(m)
            freed += w.numel() * w.element_size()
            m.quant_method = Fp8DenseLinearMethod()
            count += 1
        elif (lm_head and isinstance(m, ParallelLMHead) and w4 is not None and w4.search(name)
              and not isinstance(method, (Fp8LMHeadMethod, W4LMHeadMethod)) and w.shape[1] % 128 == 0):
            _w4()
            saved += _quantize_w4(m)
            m.quant_method = W4LMHeadMethod(m.quant_method)
            count4 += 1
        elif lm_head and isinstance(m, ParallelLMHead) and not isinstance(method, (Fp8LMHeadMethod, W4LMHeadMethod)):
            saved += _quantize(m)
            m.quant_method = Fp8LMHeadMethod(m.quant_method)
            count += 1
    torch.cuda.empty_cache()
    logger.info("Dense FP8: converted %d linears to FP8 and %d to NVFP4, %.2f GiB less to read per step",
                count, count4, saved / 2**30)


_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """w through NVFP4 (16-wide blocks along K, e4m3 block scales, one fp32
    global scale) and back to its dtype, rounding to nearest."""
    N, K = w.shape
    wf = w.float().view(N, K // 16, 16)
    gscale = wf.abs().amax().clamp(min=1e-12) / (6.0 * 448.0)
    bscale = (wf.abs().amax(-1, keepdim=True) / 6.0 / gscale).to(torch.float8_e4m3fn).float()
    bscale = torch.where(bscale == 0, torch.ones_like(bscale), bscale)
    v = wf / (bscale * gscale)
    grid = torch.tensor(_FP4_GRID, device=w.device)
    mid = (grid[1:] + grid[:-1]) / 2
    q = grid[torch.bucketize(v.abs().clamp(max=6.0), mid)] * v.sign()
    return (q * bscale * gscale).view(N, K).to(w.dtype)


@torch.no_grad()
def simulate_nvfp4(model: torch.nn.Module) -> None:
    """Diagnostic: round every eligible bf16 linear (not the lm_head) through
    NVFP4 in place, to measure what 4-bit dense weights would cost in quality."""
    if os.environ.get("VLLM_DENSE_NVFP4_SIM") != "1":
        return
    only = re.compile(os.environ.get("VLLM_DENSE_NVFP4_SIM_MATCH", "."))
    count = 0
    for name, m in model.named_modules():
        w = getattr(m, "weight", None)
        method = getattr(m, "quant_method", None)
        if not only.search(name):
            continue
        if (isinstance(m, LinearBase) and isinstance(method, UnquantizedLinearMethod)
                and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2
                and w.shape[1] % 16 == 0 and not _EXCLUDE.search(name)):
            w.copy_(_nvfp4_roundtrip(w))
            count += 1
    logger.warning("NVFP4 simulation: rounded %d linears through NVFP4", count)
