# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA speculative verify with one checkpoint state per request.

A port of Kimi-K3's RecoverSSM (models/kimi_k3/nvidia/ops/recoverssm.py and
kda_metadata.py), enabled with VLLM_GLM5NEXT_RECOVERSSM=1.

The stock spec path keeps 1 + num_spec recurrent states per request and writes
one full state per verified token. Here the verify kernel reads the request's
checkpoint, writes the outputs and records each token's raw inputs. After
sampling, the commit replays only the accepted tokens from the checkpoint.

Where this differs from Kimi-K3's version:
- The records are the raw k, gate, v and beta, and the commit replays them
  through the stock fused_recurrent_kda kernel, so the committed state is the
  one the stock spec path would have stored. Kimi-K3 stores fp32 corrections
  and folds them in closed form, which rounds differently. A copy of the stock
  recurrence also rounded differently (1-2 ulp): it read its state and records
  through integer addresses, so Triton chose other layouts and reduced sums in
  another order.
- The records live in a per-row scratch outside the KV pool: they only live
  from verify to commit within one step. Inside the mamba page they would
  make it larger than the 2304-token attention page and double the block size.
"""

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.models.glm5next.nvidia.ops.third_party.kda.fused_recurrent import (
    token_stride as _token_stride,
)
from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
)
from vllm.v1.kv_cache_interface import MambaSpec

# Tile shape and warps of the stock fused_recurrent_kda launch. The verify
# kernel uses the same so its fp32 arithmetic compiles the same way.
_BV = 8
_NUM_WARPS = 1
_NUM_STAGES = 3


@triton.jit
def _verify_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    rec_kv,
    rec_g,
    rec_beta,
    cu_seqlens,
    state_indices,
    a_log,
    g_bias,
    scale,
    stride_q_t,
    stride_k_t,
    stride_v_t,
    stride_beta_t,
    stride_state_indices,
    stride_state: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    Q: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    # The per-token body is fused_recurrent_gated_delta_rule_fwd_kernel's
    # (IS_KDA, COMPUTE_GATE, SIGMOID_BETA, USE_QK_L2NORM_IN_KERNEL). Keep the
    # two in step: the equivalence test compares them bit for bit.
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q_t + i_h * K + o_k
    p_k = k + bos * stride_k_t + i_h * K + o_k
    p_v = v + bos * stride_v_t + i_h * V + o_v
    p_beta = beta + bos * stride_beta_t + i_h
    p_gk = g + (bos * H + i_h) * K + o_k
    b_a_log = tl.exp(tl.load(a_log + i_h).to(tl.float32))
    p_o = o + (bos * H + i_h) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(state_indices + i_n * stride_state_indices).to(tl.int64)
    if state_idx <= 0:
        return
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    p_h0 = h0 + state_idx * stride_state + i_h * V * K + o_v[:, None] * K + o_k[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Row i_n owns record tokens [i_n * Q, i_n * Q + Q), laid out the way the
    # stock kernel reads q/k/v, g and beta. The commit reads them in the same
    # step over the same rows, so they never outlive a step.
    tok = i_n * Q
    p_rk = rec_kv + (tok * 2 * H + i_h) * K + o_k
    p_rv = rec_kv + ((tok * 2 + 1) * H + i_h) * V + o_v
    p_rg = rec_g + (tok * H + i_h) * K + o_k
    p_rbeta = rec_beta + tok * H + i_h

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        r_k = tl.load(p_k, mask=mask_k, other=0)
        b_k = r_k.to(tl.float32)
        r_v = tl.load(p_v, mask=mask_v, other=0)
        b_v = r_v.to(tl.float32)

        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        r_g = tl.load(p_gk)
        b_gk = r_g.to(tl.float32)
        b_gk += tl.load(g_bias + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
        b_h *= exp(b_gk[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        r_beta = tl.load(p_beta)
        b_beta = r_beta.to(tl.float32)
        b_beta = tl.sigmoid(b_beta)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        tl.store(p_rv + i_t * 2 * H * V, r_v, mask=mask_v)
        if i_v == 0:
            tl.store(p_rk + i_t * 2 * H * K, r_k, mask=mask_k)
            tl.store(p_rg + i_t * H * K, r_g, mask=mask_k)
            tl.store(p_rbeta + i_t * H, r_beta)

        p_q += stride_q_t
        p_k += stride_k_t
        p_o += H * V
        p_v += stride_v_t
        p_gk += H * K
        p_beta += stride_beta_t


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    # Kimi-K3's RecoverSSM, except for the final state's column. Kimi-K3 uses
    # n // block_size, where n is the committed count. When n ends exactly on a
    # boundary, that is the next block, which align mode has not allocated
    # yet, so the state was lost. This uses (n - 1) // block_size, the block
    # holding the last token, as the stock align path does; the boundary block
    # is then the final block itself. _record_state_block_kernel records the
    # same column for the next step's pre-step copy.
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
            tl.int32
        )
        final_num_computed = num_computed + commit_len
        final_state_col = tl.minimum(
            tl.maximum(final_num_computed - 1, 0) // mamba_block_size,
            block_table_width - 1,
        )
        final_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )


@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    # Unchanged from Kimi-K3's RecoverSSM. After the verify the conv state is
    # [history[1:], x_0 .. x_{q-1}]; the window after n accepted tokens starts
    # at n - 1, and moves to column 0 so the next step reads it with
    # num_accepted_tokens == 1.
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
    dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
    token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = base_addr.to(tl.pointer_type(conv_state_ref_ptr.dtype.element_ty))
    source_ptr = conv_state_ptr + source_state_idx * block_stride
    final_ptr = conv_state_ptr + final_state_idx * block_stride

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_HISTORY)
    mask = (offs_d[:, None] < conv_dim) & (offs_h[None, :] < conv_history_len)
    final_values = tl.load(
        source_ptr
        + offs_d[:, None] * dim_stride
        + (commit_len - 1 + offs_h[None, :]) * token_stride,
        mask=mask,
    )
    if ALIGN_MODE:
        boundary_values = tl.load(
            source_ptr
            + offs_d[:, None] * dim_stride
            + (boundary_recovery_len - 1 + offs_h[None, :]) * token_stride,
            mask=mask & (boundary_state_idx > null_block_id),
        )
        boundary_ptr = conv_state_ptr + boundary_state_idx * block_stride
        tl.store(
            boundary_ptr
            + offs_d[:, None] * dim_stride
            + offs_h[None, :] * token_stride,
            boundary_values,
            mask=mask & (boundary_state_idx > null_block_id),
        )
    tl.store(
        final_ptr + offs_d[:, None] * dim_stride + offs_h[None, :] * token_stride,
        final_values,
        mask=mask,
    )


@triton.jit
def _replay_indices_kernel(
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    replay_indices_ptr,
    null_block_id,
    stride_state_indices,
    Q: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One row of the stock kernel's ssm_state_indices: it reads the initial
    # state at column num_accepted - 1 (Q here) and stores the state after
    # token t to column t when that is not the null block.
    i_n = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    source = tl.load(state_indices_ptr + i_n * stride_state_indices).to(tl.int32)
    commit_len = tl.load(commit_lens_ptr + i_n)
    final = tl.load(final_state_indices_ptr + i_n).to(tl.int32)
    boundary = tl.load(boundary_state_indices_ptr + i_n).to(tl.int32)
    boundary_len = tl.load(boundary_recovery_lens_ptr + i_n)
    row = tl.where(cols == commit_len - 1, final, null_block_id)
    # A boundary at the last token is the final state's own block.
    at_boundary = (
        (cols == boundary_len - 1)
        & (boundary_len < commit_len)
        & (boundary > null_block_id)
    )
    row = tl.where(at_boundary, boundary, row)
    row = tl.where(cols == Q, source, row)
    valid = (source > null_block_id) & (commit_len > 0) & (final > null_block_id)
    row = tl.where(valid, row, null_block_id)
    tl.store(replay_indices_ptr + i_n * (Q + 1) + cols, row, mask=cols < Q + 1)


@dataclass
class Records:
    """One layer's verify inputs for up to ``rows`` rows of ``q`` tokens.

    Token-major so the stock recurrent kernel reads them as q/k/v, g, beta:
    ``kv[t, 0]`` is k and ``kv[t, 1]`` is v for record token ``t``.
    """

    kv: torch.Tensor  # [rows * q, 2, heads, head_dim]
    g: torch.Tensor  # [rows * q, heads, head_dim]
    beta: torch.Tensor  # [rows * q, heads]
    rows: int
    q: int


def allocate_records(
    num_rows: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Records:
    tokens = num_rows * spec_query_len
    return Records(
        kv=torch.empty((tokens, 2, num_heads, head_dim), dtype=dtype, device=device),
        g=torch.empty((tokens, num_heads, head_dim), dtype=dtype, device=device),
        beta=torch.empty((tokens, num_heads), dtype=dtype, device=device),
        rows=num_rows,
        q=spec_query_len,
    )


def kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    lower_bound: float,
    checkpoint_state: torch.Tensor,
    records: Records,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a speculative window from each row's checkpoint without writing it.

    Takes what kda.py passes fused_recurrent_kda on its spec path (raw gate
    logits and raw beta), with ``state_indices`` holding one checkpoint block
    per row. Writes the outputs and fills ``records`` for the commit.
    """
    _, total_tokens, num_heads, key_dim = k.shape
    value_dim = v.shape[-1]
    num_rows = state_indices.shape[0]
    if query_start_loc.shape[0] != num_rows + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
    if num_rows > records.rows:
        raise ValueError("KDA RecoverSSM batch exceeds its record rows")
    if records.kv.shape[1:] != (2, num_heads, key_dim) or value_dim != key_dim:
        raise ValueError("KDA RecoverSSM records do not match the layer")
    if records.kv.dtype != k.dtype or records.beta.dtype != beta.dtype:
        raise ValueError("KDA RecoverSSM records must use the activation dtype")
    # No token-count check: FULL graphs pad the token count, and the builder
    # already routes rows longer than the record window to the prefill path.
    if out is None:
        out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    else:
        assert out.shape == k.shape and out.dtype == k.dtype
        assert out.is_contiguous()
    if total_tokens == 0 or num_rows == 0:
        return out

    g = g.contiguous()
    BK = triton.next_power_of_2(key_dim)
    BV = min(triton.next_power_of_2(value_dim), _BV)
    assert BK == key_dim, "KDA RecoverSSM needs a power-of-two head dim"
    grid = (1, triton.cdiv(value_dim, BV), num_rows * num_heads)
    _verify_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        out,
        checkpoint_state,
        records.kv,
        records.g,
        records.beta,
        query_start_loc,
        state_indices,
        a_log.reshape(-1).contiguous(),
        g_bias.reshape(-1).contiguous(),
        key_dim**-0.5,
        _token_stride(q),
        _token_stride(k),
        _token_stride(v),
        _token_stride(beta),
        state_indices.stride(0),
        stride_state=checkpoint_state.stride(0),
        H=num_heads,
        K=key_dim,
        V=value_dim,
        Q=records.q,
        BK=BK,
        BV=BV,
        LOWER_BOUND=lower_bound,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out


@dataclass
class Glm5NextRecoverSSMCommitContext:
    """Per-KV-group commit state: every layer's pages, records and gate params."""

    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    layers: tuple[Any, ...]
    lower_bound: float
    spec_query_len: int
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    replay_indices: torch.Tensor
    replay_query_start_loc: torch.Tensor
    replay_num_accepted: torch.Tensor
    replay_out: torch.Tensor

    @classmethod
    def create(
        cls, layers: list[Any], *, spec_query_len: int, max_num_reqs: int
    ) -> "Glm5NextRecoverSSMCommitContext":
        """``layers``: KDA layers with ``kv_cache`` (conv, state), ``A_log``,
        ``dt_bias``, ``kda_lower_bound`` and ``recoverssm_records``."""
        if not layers:
            raise ValueError("KDA RecoverSSM commit requires at least one layer")
        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [state.transpose(-1, -2) for state in conv_states]
        checkpoints = [layer.kv_cache[1] for layer in layers]
        lower_bounds = {layer.kda_lower_bound for layer in layers}
        if len(lower_bounds) != 1:
            raise ValueError("KDA RecoverSSM layers need matching gate bounds")

        state_ref = checkpoints[0]
        _, num_heads, value_dim, key_dim = state_ref.shape
        for state in checkpoints:
            if (
                state.shape[1:] != state_ref.shape[1:]
                or state.dtype != state_ref.dtype
                or state.stride()[1:] != (value_dim * key_dim, key_dim, 1)
            ):
                raise ValueError("KDA RecoverSSM layers need matching state layout")
        record_rows = layers[0].recoverssm_records.rows
        for layer in layers:
            records = layer.recoverssm_records
            if records.q != spec_query_len or records.rows != record_rows:
                raise ValueError("KDA RecoverSSM layers need matching records")
            if records.rows < max_num_reqs:
                raise ValueError("KDA RecoverSSM records do not cover max_num_seqs")
            if records.kv.shape[1:] != (2, num_heads, key_dim):
                raise ValueError("KDA RecoverSSM records do not match the state")

        conv_ref = conv_states[0]
        conv_history_len = conv_ref.shape[2] - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("KDA RecoverSSM conv state is shorter than its window")
        for conv_state in conv_states:
            if conv_state.shape[1:] != conv_ref.shape[1:] or (
                conv_state.dtype != conv_ref.dtype
            ):
                raise ValueError("KDA RecoverSSM layers need matching conv state")

        device = state_ref.device
        rows = max_num_reqs

        def _i64(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=device)

        def _i32(n: int) -> torch.Tensor:
            return torch.empty(n, dtype=torch.int32, device=device)

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=_i64([s.data_ptr() for s in conv_states]),
            conv_state_block_strides=_i64([s.stride(0) for s in conv_states]),
            conv_state_dim_strides=_i64([s.stride(1) for s in conv_states]),
            conv_state_token_strides=_i64([s.stride(2) for s in conv_states]),
            conv_history_len=conv_history_len,
            layers=tuple(layers),
            lower_bound=lower_bounds.pop(),
            spec_query_len=spec_query_len,
            commit_lens=_i32(rows),
            final_state_indices=_i32(rows),
            boundary_state_indices=_i32(rows),
            boundary_recovery_lens=_i32(rows),
            replay_indices=torch.zeros(
                (rows, spec_query_len + 1), dtype=torch.int32, device=device
            ),
            replay_query_start_loc=torch.arange(
                0,
                (rows + 1) * spec_query_len,
                spec_query_len,
                dtype=torch.int32,
                device=device,
            ),
            replay_num_accepted=torch.full(
                (rows,), spec_query_len + 1, dtype=torch.int32, device=device
            ),
            # Shaped like the records' k, as the stock kernel requires.
            replay_out=torch.empty(
                (1, record_rows * spec_query_len, num_heads, value_dim),
                dtype=layers[0].recoverssm_records.kv.dtype,
                device=device,
            ),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Fold each row's accepted tokens into its conv and recurrent state.

        ``num_accepted_tokens`` is per batch row (the runner's num_sampled);
        ``request_indices`` maps spec rows to batch rows when they differ.
        """
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("KDA RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA RecoverSSM commit metadata is incompatible")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(a is not None for a in align_args) and any(
            a is None for a in align_args
        ):
            raise ValueError("KDA RecoverSSM align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError(
                "KDA RecoverSSM align block size must cover one speculative window"
            )
        align = block_table is not None
        block_table_stride = block_table.stride() if align else (0, 0)

        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if align else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_tokens.stride(0) if align else 0,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )

        num_layers = len(self.layers)
        conv_dim = self.conv_states[0].shape[1]
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
            self.conv_states[0],
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            ALIGN_MODE=align,
            num_warps=4,
        )

        q_len = self.spec_query_len
        _replay_indices_kernel[(batch,)](
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            self.replay_indices,
            NULL_BLOCK_ID,
            state_indices.stride(0),
            Q=q_len,
            BLOCK=triton.next_power_of_2(q_len + 1),
            num_warps=1,
        )
        # The stock kernel runs all Q record tokens of each row and writes a
        # state only to the columns set above. Same kernel, same inputs, so
        # those states match the stock spec path's bit for bit.
        cu_seqlens = self.replay_query_start_loc[: batch + 1]
        indices = self.replay_indices[:batch]
        num_accepted = self.replay_num_accepted[:batch]
        for layer in self.layers:
            records = layer.recoverssm_records
            k = records.kv[:, 0].unsqueeze(0)
            fused_recurrent_kda(
                q=k,
                k=k,
                v=records.kv[:, 1].unsqueeze(0),
                g=records.g.unsqueeze(0),
                beta=records.beta.unsqueeze(0),
                initial_state=layer.kv_cache[1],
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=indices,
                num_accepted_tokens=num_accepted,
                out=self.replay_out,
                sigmoid_beta=True,
                a_log=layer.A_log,
                g_bias=layer.dt_bias,
                compute_gate=True,
                lower_bound=self.lower_bound,
            )


@dataclass
class _AlignCommit:
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int


@dataclass
class _Commit:
    state_indices: torch.Tensor
    query_start_loc: torch.Tensor
    request_indices: torch.Tensor | None
    align: _AlignCommit | None


@dataclass
class Glm5NextRecoverSSMMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    recoverssm_commit: _Commit | None = None
    recoverssm_context: Glm5NextRecoverSSMCommitContext | None = field(
        default=None, repr=False, compare=False
    )

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        commit = self.recoverssm_commit
        if commit is None:
            return None
        assert self.recoverssm_context is not None
        align = commit.align
        self.recoverssm_context.commit(
            num_accepted_tokens,
            commit.state_indices[: self.num_spec_decodes, 0],
            commit.query_start_loc[: self.num_spec_decodes + 1],
            request_indices=commit.request_indices,
            block_table=align.block_table if align is not None else None,
            num_computed_tokens=(
                align.num_computed_tokens if align is not None else None
            ),
            mamba_block_size=align.block_size if align is not None else None,
        )
        if align is None:
            return None
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=self.num_spec_decodes,
            request_indices=commit.request_indices,
            block_table=align.block_table,
            num_computed_tokens=align.num_computed_tokens,
            block_size=align.block_size,
        )


def _check_config(vllm_config: VllmConfig) -> None:
    """Fail at startup where the commit would otherwise be skipped silently."""
    from vllm.models.glm5next.common import model as glm_model

    if not getattr(glm_model, "_RECOVERSSM", False):
        raise RuntimeError(
            "VLLM_GLM5NEXT_RECOVERSSM=1 needs experimental/fixes/model.py "
            "mounted (recoverssm.yaml or sp.yaml): it installs the commit hook"
        )
    if not vllm_config.use_v2_model_runner:
        raise RuntimeError("KDA RecoverSSM commits from the V2 model runner only")
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise RuntimeError("KDA RecoverSSM requires pipeline_parallel_size=1")
    if vllm_config.cache_config.mamba_cache_mode not in ("none", "align"):
        raise RuntimeError("KDA RecoverSSM supports mamba cache modes none and align")


class Glm5NextRecoverSSMMetadataBuilder(GDNAttentionMetadataBuilder):
    """GDN metadata for GLM's KDA with one state slot per spec-decode row.

    Every running decode row goes down the spec path, drafts or not, so the
    conv state is always in the compacted form the commit leaves. Steps
    without such rows use the GDN builder unchanged.
    """

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        _check_config(vllm_config)
        assert self.use_spec_decode, "KDA RecoverSSM needs speculative decoding"
        self.spec_query_len = self.num_spec + 1
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        rows = max(self.decode_cudagraph_max_bs, self.max_num_reqs)
        self.rs_state_indices = torch.empty((rows, 1), dtype=torch.int32, device=device)
        # The conv kernel reads the window at num_accepted - 1; the commit
        # already moved it to column 0.
        self.rs_ones = torch.ones(rows, dtype=torch.int32, device=device)
        self.recoverssm_context: Glm5NextRecoverSSMCommitContext | None = None
        # Allocated here, before any capture: the eager KDA segment must not
        # allocate during a graph capture.
        layers = vllm_config.compilation_config.static_forward_context
        for name in layer_names:
            layer = layers[name]
            if getattr(layer, "recoverssm_records", None) is None:
                layer.recoverssm_records = allocate_records(
                    self.max_num_reqs,
                    layer.local_num_heads,
                    layer.head_dim,
                    self.spec_query_len,
                    vllm_config.model_config.dtype,
                    device,
                )

    def _get_context(self) -> Glm5NextRecoverSSMCommitContext:
        if self.recoverssm_context is None:
            layers = self.vllm_config.compilation_config.static_forward_context
            self.recoverssm_context = Glm5NextRecoverSSMCommitContext.create(
                [layers[name] for name in self.layer_names],
                spec_query_len=self.spec_query_len,
                max_num_reqs=self.max_num_reqs,
            )
        return self.recoverssm_context

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata
        if num_decode_draft_tokens_cpu is None:
            return super().build(
                common_prefix_len, m, num_accepted_tokens, None, fast_build
            )
        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        query_lens_cpu = query_start_loc_cpu.diff()
        spec_mask_cpu = num_decode_draft_tokens_cpu >= 0
        if m.is_prefilling is not None:
            spec_mask_cpu |= (~m.is_prefilling) & (query_lens_cpu > 0)
        # Rows longer than the record window (adaptive probes) run as prefills.
        spec_mask_cpu &= query_lens_cpu <= self.spec_query_len
        num_spec_decodes = int(spec_mask_cpu.sum())
        if num_spec_decodes == 0:
            return super().build(
                common_prefix_len, m, num_accepted_tokens, None, fast_build
            )

        block_table = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        device = query_start_loc.device
        num_query_tokens = int(query_start_loc_cpu[-1])
        active_non_spec_cpu = (~spec_mask_cpu) & (query_lens_cpu > 0)
        num_prefills = int(active_non_spec_cpu.sum())
        num_prefill_tokens = int(query_lens_cpu[active_non_spec_cpu].sum())
        num_spec_decode_tokens = num_query_tokens - num_prefill_tokens

        request_indices = None
        has_initial_state = None
        nums_dict = batch_ptr = token_chunk_offset_ptr = None
        if num_prefills == 0:
            # Real rows precede the cudagraph padding, and all of them verify.
            spec_token_indx = non_spec_token_indx = None
            spec_state_indices = block_table[:num_spec_decodes, :1]
            non_spec_state_indices = None
            spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
            non_spec_query_start_loc = None
        else:
            query_lens = query_start_loc.diff()
            spec_mask = async_tensor_h2d(spec_mask_cpu, device=device)
            request_indices = async_tensor_h2d(
                spec_mask_cpu.nonzero(as_tuple=True)[0],
                dtype=torch.int32,
                device=device,
            )
            spec_token_masks = torch.repeat_interleave(
                spec_mask, query_lens, output_size=num_query_tokens
            )
            # Stable, so each row's tokens stay in order in both halves.
            index = torch.argsort(spec_token_masks, stable=True)
            non_spec_token_indx = index[:num_prefill_tokens]
            spec_token_indx = index[num_prefill_tokens:]
            spec_state_indices = block_table[spec_mask_cpu, :1]
            non_spec_state_indices = block_table[active_non_spec_cpu, 0]
            spec_query_start_loc = torch.zeros(
                num_spec_decodes + 1, dtype=torch.int32, device=device
            )
            torch.cumsum(query_lens[spec_mask_cpu], dim=0, out=spec_query_start_loc[1:])
            non_spec_query_start_loc = torch.zeros(
                num_prefills + 1, dtype=torch.int32, device=device
            )
            torch.cumsum(
                query_lens[active_non_spec_cpu],
                dim=0,
                out=non_spec_query_start_loc[1:],
            )
            non_spec_query_start_loc_cpu = torch.zeros(
                num_prefills + 1, dtype=torch.int32
            )
            torch.cumsum(
                query_lens_cpu[active_non_spec_cpu],
                dim=0,
                out=non_spec_query_start_loc_cpu[1:],
            )
            has_initial_state = (m.compute_num_computed_tokens() > 0)[
                active_non_spec_cpu
            ]
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu, device=device
                )
            )
        num_accepted = self.rs_ones[:num_spec_decodes]

        batch_size = m.num_reqs
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and batch_size <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            self.rs_state_indices[:num_spec_decodes].copy_(
                spec_state_indices, non_blocking=True
            )
            spec_state_indices = self.rs_state_indices[:batch_size]
            spec_state_indices[num_spec_decodes:].fill_(NULL_BLOCK_ID)
            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)
            num_accepted = self.rs_ones[:batch_size]

        align = None
        if self.kv_cache_spec.mamba_cache_mode == "align":
            align = _AlignCommit(
                block_table=m.block_table_tensor,
                num_computed_tokens=m.compute_num_computed_tokens(),
                block_size=self.kv_cache_spec.block_size,
            )
        return Glm5NextRecoverSSMMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices,
            non_spec_state_indices_tensor=non_spec_state_indices,
            spec_sequence_masks=None,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            recoverssm_commit=_Commit(
                state_indices=spec_state_indices,
                query_start_loc=spec_query_start_loc,
                request_indices=request_indices,
                align=align,
            ),
            recoverssm_context=self._get_context(),
        )


class Glm5NextRecoverSSMBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_KDA_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[Glm5NextRecoverSSMMetadataBuilder]:
        return Glm5NextRecoverSSMMetadataBuilder


@triton.heuristics(
    {"HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None}
)
@triton.jit
def _record_state_block_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    request_indices_ptr,
    num_computed_ptr,
    state_idx_ptr,
    num_accepted_ptr,
    HAS_REQUEST_INDICES: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WIDTH: tl.constexpr,
):
    # The runner's _postprocess_recoverssm_align_kernel, except for the column:
    # the commit left the state in the block holding the last committed token.
    spec_idx = tl.program_id(0)
    batch_idx = spec_idx
    if HAS_REQUEST_INDICES:
        batch_idx = tl.load(request_indices_ptr + spec_idx)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_computed = tl.load(num_computed_ptr + batch_idx)
    last = tl.maximum(num_computed + num_sampled - 1, 0)
    tl.store(
        state_idx_ptr + req_state_idx,
        tl.minimum(last // MAMBA_BLOCK_SIZE, BLOCK_TABLE_WIDTH - 1),
    )
    tl.store(num_accepted_ptr + req_state_idx, 1)


def record_state_blocks(
    meta: RecoverSSMPostprocessMetadata,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Point each committed request's running state column at its state block.

    The next step's pre-step copy moves it from there when the step's tokens
    pass into the next block.
    """
    _record_state_block_kernel[(meta.num_spec_decodes,)](
        idx_mapping,
        num_sampled,
        meta.request_indices,
        meta.num_computed_tokens,
        state_indices,
        num_accepted_tokens,
        MAMBA_BLOCK_SIZE=meta.block_size,
        BLOCK_TABLE_WIDTH=meta.block_table.shape[1],
    )


@cache
def model_state_cls() -> type:
    """MambaHybridModelState with the runner's RecoverSSM commit hook.

    The stock state creates the hook only when cache_config.use_kda_recoverssm
    is set, which VllmConfig's validator allows for Kimi-K3 alone and resets
    on every re-validation (the DFlash loader re-validates after the target
    model loads). The hook records the state block with record_state_blocks.
    """
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

    class Glm5NextRecoverSSMState(RecoverSSMState):
        def commit_step(
            self,
            num_sampled: torch.Tensor | int,
            idx_mapping: torch.Tensor,
            *,
            state_indices: torch.Tensor | None,
            num_accepted_tokens: torch.Tensor,
        ) -> None:
            step = self._step
            self._step = None
            if isinstance(num_sampled, int) or step is None:
                return
            for metadata in step:
                meta = metadata.commit_recoverssm_state(num_sampled)
                if meta is None:
                    continue
                assert state_indices is not None
                record_state_blocks(
                    meta, idx_mapping, num_sampled, state_indices, num_accepted_tokens
                )

    class Glm5NextRecoverSSMModelState(MambaHybridModelState):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.recoverssm = Glm5NextRecoverSSMState()

    return Glm5NextRecoverSSMModelState
