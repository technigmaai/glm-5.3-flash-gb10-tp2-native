"""Sparse NoPE MLA attention for GB10 in Triton: each query token attends to
its own top-k KV rows (valid prefix of `idx`, -1 past it).

One program per query token: all heads' rows of Q stay in registers, and the
selected KV rows stream through in blocks, gathered by index. K and V are the
same latent row (NoPE MLA), FP8 E4M3 with one tensor-wide scale.

Small batches (decode: 1 + drafts tokens per request) leave most of the 48 SMs
idle with one program per token, so they split each token's rows across
SPLITS programs, which write partial softmax state that a second kernel
combines (flash-decoding).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_mla_split(q_ptr, kv_ptr, idx_ptr, len_ptr, acc_ptr, ml_ptr, qk_scale, sq_t, sq_h,
                      H: tl.constexpr, D: tl.constexpr, TOPK: tl.constexpr, CHUNK: tl.constexpr,
                      SPLITS: tl.constexpr, BLOCK_N: tl.constexpr, HP: tl.constexpr):
    t = tl.program_id(0)
    sp = tl.program_id(1)
    heads = tl.arange(0, HP)
    hm = heads < H
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + t * sq_t + heads[:, None] * sq_h + dims[None, :], mask=hm[:, None], other=0.0).to(tl.float16)
    n_valid = tl.load(len_ptr + t)
    m_i = tl.full([HP], float("-inf"), tl.float32)
    l_i = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for start in range(sp * CHUNK, sp * CHUNK + CHUNK, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        live = offs < n_valid
        rows = tl.load(idx_ptr + t * TOPK + offs, mask=live, other=0)
        kv = tl.load(kv_ptr + rows[:, None].to(tl.int64) * D + dims[None, :], mask=live[:, None], other=0.0)
        kv = kv.to(tl.float16)
        s = tl.dot(q, tl.trans(kv)) * qk_scale
        s = tl.where(live[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))
        p = tl.where(m_new[:, None] == float("-inf"), 0.0, tl.exp2(s - m_new[:, None]))
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), kv)
        m_i = m_new
    base = (t * SPLITS + sp) * H
    tl.store(acc_ptr + (base + heads[:, None]) * D + dims[None, :], acc, mask=hm[:, None])
    tl.store(ml_ptr + (base + heads) * 2, m_i, mask=hm)
    tl.store(ml_ptr + (base + heads) * 2 + 1, l_i, mask=hm)


@triton.jit
def _sparse_mla_combine(acc_ptr, ml_ptr, o_ptr, kv_scale, H: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    dims = tl.arange(0, D)
    sps = tl.arange(0, SPLITS)
    base = (t * SPLITS + sps) * H + h
    m = tl.load(ml_ptr + base * 2)
    l = tl.load(ml_ptr + base * 2 + 1)
    m_max = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - m_max))
    l_sum = tl.sum(l * w, 0)
    acc = tl.load(acc_ptr + base[:, None] * D + dims[None, :])
    out = tl.sum(acc * w[:, None], 0)
    out = tl.where(l_sum > 0, out / l_sum, 0.0) * kv_scale
    tl.store(o_ptr + (t * H + h) * D + dims, out.to(o_ptr.dtype.element_ty))


@triton.jit
def _sparse_mla_fwd(q_ptr, kv_ptr, idx_ptr, len_ptr, o_ptr, qk_scale, kv_scale, sq_t, sq_h,
                    H: tl.constexpr, D: tl.constexpr, TOPK: tl.constexpr, BLOCK_N: tl.constexpr, HP: tl.constexpr):
    t = tl.program_id(0)
    heads = tl.arange(0, HP)
    hm = heads < H
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + t * sq_t + heads[:, None] * sq_h + dims[None, :], mask=hm[:, None], other=0.0).to(tl.float16)
    n_valid = tl.load(len_ptr + t)
    m_i = tl.full([HP], float("-inf"), tl.float32)
    l_i = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for start in range(0, TOPK, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        live = offs < n_valid
        rows = tl.load(idx_ptr + t * TOPK + offs, mask=live, other=0)
        kv = tl.load(kv_ptr + rows[:, None].to(tl.int64) * D + dims[None, :], mask=live[:, None], other=0.0)
        kv = kv.to(tl.float16)
        s = tl.dot(q, tl.trans(kv)) * qk_scale                 # [H, BLOCK_N], log2 domain
        s = tl.where(live[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), kv)
        m_i = m_new
    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0) * kv_scale  # a query with no KV rows gives 0
    tl.store(o_ptr + t * H * D + heads[:, None] * D + dims[None, :], out.to(o_ptr.dtype.element_ty), mask=hm[:, None])


# Programs a split launch aims for: two per SM on GB10's 48.
_SPLIT_TARGET = 96


def sparse_mla(q, kv, idx, lens, sm_scale, kv_scale=1.0, block_n=32, num_warps=4, num_stages=2, splits=None):
    """q [T, H, D] bf16 (last dim contiguous), kv [S, D] fp8 e4m3, idx [T, TOPK]
    int32 global rows with the valid ones first, lens [T] int32 valid counts.
    Returns [T, H, D] bf16. The 1.44 folds exp into exp2."""
    T, H, D = q.shape
    # TP=3: 22 or 24 heads/rank. Tile at the next power of two (>=16 for tl.dot), masked.
    hp = max(16, triton.next_power_of_2(H))
    assert q.stride(2) == 1 and kv.is_contiguous() and idx.stride(1) == 1
    out = torch.empty(T, H, D, dtype=q.dtype, device=q.device)
    qk_scale = sm_scale * kv_scale * 1.4426950408889634
    topk = idx.shape[1]
    if splits is None:
        splits = 1
        while T * splits < _SPLIT_TARGET and splits < 16 and topk % (splits * 2 * block_n) == 0:
            splits *= 2
    if splits > 1:
        acc = torch.empty(T, splits, H, D, dtype=torch.float32, device=q.device)
        ml = torch.empty(T, splits, H, 2, dtype=torch.float32, device=q.device)
        _sparse_mla_split[(T, splits)](q, kv, idx, lens, acc, ml, qk_scale, q.stride(0), q.stride(1),
                                       H=H, D=D, TOPK=topk, CHUNK=topk // splits, SPLITS=splits,
                                       BLOCK_N=block_n, HP=hp, num_warps=num_warps, num_stages=num_stages)
        _sparse_mla_combine[(T, H)](acc, ml, out, kv_scale, H=H, D=D, SPLITS=splits, num_warps=4)
        return out
    _sparse_mla_fwd[(T,)](q, kv, idx, lens, out, qk_scale, kv_scale, q.stride(0), q.stride(1),
                          H=H, D=D, TOPK=idx.shape[1],
                          BLOCK_N=block_n, HP=hp, num_warps=num_warps, num_stages=num_stages)
    return out
