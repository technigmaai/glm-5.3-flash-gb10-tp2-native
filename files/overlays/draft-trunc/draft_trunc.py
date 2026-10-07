"""Per-step draft truncation.

The scheduler picks how many drafts each request verifies before the drafts
exist, so the verify shape is fixed. Truncation leaves the shape alone and
marks a request's later drafts dead on the device: their MoE routes go to
expert -1, which the MoE kernels skip without reading those experts' weights,
and the rejection sampler sees them as -1, so acceptance stops before them.
They come after the live tokens, so causal attention leaves the live tokens
exact.

By default (VLLM_DRAFT_TRUNC_TAU, 0.3 in adaptive-k.yaml) each request keeps
its leading drafts whose predicted survival, the running product of the
acceptance estimator's per-draft probabilities, is at least tau. A cut step
grades only the drafts the estimator already expected to survive, so fitting
on it would bias the estimator, and a position it always kills could never
recover. Every VLLM_DRAFT_TRUNC_EXPLORE-th step (4 by default) therefore skips
the cut, and only those steps are folded into the estimator. The file named by
VLLM_DRAFT_TRUNC_CONTROL overrides that while it exists: "tau X" sets the
threshold, "live N" keeps each request's first N drafts, and anything else
truncates nothing.
"""
import os

import torch
import vllm

from vllm.triton_utils import tl, triton

# The other files in this directory are whole copies of vLLM's, with the cut added. They
# match one vLLM commit, the image's VLLM_REF. On any other they would put back old code
# over new, so the engine stops instead.
VLLM_COMMIT = "ddd6fbca"
if f"+g{VLLM_COMMIT}" not in vllm.__version__:
    raise RuntimeError(f"experimental/draft-trunc copies vLLM {VLLM_COMMIT}'s files, but this image runs "
                       f"vLLM {vllm.__version__}. Refresh the copies from this image, or remove the "
                       "draft-trunc mounts from experimental/compose/adaptive-k.yaml.")

_TAU = float(os.environ.get("VLLM_DRAFT_TRUNC_TAU") or 0)
ENABLED = bool(os.environ.get("VLLM_DRAFT_TRUNC_CONTROL")) or _TAU > 0
# Measurement: per verify step, each request's draft confidences (the
# acceptance estimator's, computed when the drafts were made) and how many
# tokens it sampled, one JSON line per step, on tensor-parallel rank 0.
LOG = os.environ.get("VLLM_DRAFT_TRUNC_LOG", "")
_LOG_MAX = int(os.environ.get("VLLM_DRAFT_TRUNC_LOG_STEPS", "20000"))
_logged = 0
_control = os.environ.get("VLLM_DRAFT_TRUNC_CONTROL", "")
_mtime = 0.0
_mode: tuple[str, float] | None = None
_EXPLORE = int(os.environ.get("VLLM_DRAFT_TRUNC_EXPLORE") or 4)
_steps = 0
_cut = False


def _read_control() -> tuple[str, float] | None:
    """("live", N), ("tau", X), or None to truncate nothing."""
    global _mtime, _mode
    default = ("tau", _TAU) if _TAU > 0 else None
    try:
        mtime = os.stat(_control).st_mtime if _control else None
    except OSError:
        mtime = None
    if mtime is None:
        _mtime, _mode = 0.0, default
        return default
    if mtime != _mtime:
        _mtime = mtime
        words = open(_control).read().split()
        _mode = (words[0], float(words[1])) if len(words) == 2 and words[0] in ("live", "tau") else None
    return _mode


def live_drafts() -> int | None:
    """The control file's N, or None."""
    mode = _read_control()
    return int(mode[1]) if mode and mode[0] == "live" else None


@triton.jit
def _mark_dead_kernel(is_padding_ptr, query_start_loc_ptr, cu_num_logits_ptr, live, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    num_drafts = tl.load(cu_num_logits_ptr + b + 1) - tl.load(cu_num_logits_ptr + b) - 1
    end = tl.load(query_start_loc_ptr + b + 1)
    j = tl.arange(0, BLOCK)
    # A request's drafts are its last num_drafts tokens.
    tl.store(is_padding_ptr + end - num_drafts + j, tl.full((BLOCK,), 1, tl.int1),
             mask=(j >= live) & (j < num_drafts))


@triton.jit
def _mark_cut_kernel(is_padding_ptr, query_start_loc_ptr, cu_num_logits_ptr, pred_ptr, pred_stride,
                     idx_mapping_ptr, tau, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    num_drafts = tl.load(cu_num_logits_ptr + b + 1) - tl.load(cu_num_logits_ptr + b) - 1
    end = tl.load(query_start_loc_ptr + b + 1)
    state = tl.load(idx_mapping_ptr + b).to(tl.int64)
    j = tl.arange(0, BLOCK)
    in_range = j < num_drafts
    p = tl.load(pred_ptr + state * pred_stride + j, mask=in_range & (state >= 0), other=1.0)
    # Survival only falls along a request, so the dead drafts are a suffix.
    dead = (tl.cumprod(p, axis=0) < tau) & in_range
    tl.store(is_padding_ptr + end - num_drafts + j, tl.full((BLOCK,), 1, tl.int1), mask=dead)


def mark_dead(is_padding: torch.Tensor, query_start_loc: torch.Tensor, cu_num_logits: torch.Tensor,
              num_reqs: int, max_drafts: int, predictions: torch.Tensor | None = None,
              idx_mapping: torch.Tensor | None = None) -> None:
    """Set is_padding on each request's dead drafts, per the control file."""
    global _steps, _cut
    _cut = False
    mode = _read_control()
    if mode is None or num_reqs == 0 or max_drafts == 0:
        return
    block = triton.next_power_of_2(max_drafts)
    if mode[0] == "live":
        _mark_dead_kernel[(num_reqs,)](is_padding, query_start_loc, cu_num_logits, int(mode[1]), BLOCK=block)
        _cut = True
        return
    if predictions is None or idx_mapping is None:
        return
    _steps += 1
    if _EXPLORE > 0 and _steps % _EXPLORE == 0:
        return
    _mark_cut_kernel[(num_reqs,)](is_padding, query_start_loc, cu_num_logits, predictions, predictions.stride(0),
                                  idx_mapping, mode[1], BLOCK=block)
    _cut = True


def graded() -> bool:
    """Whether the last step verified every draft, so the estimator may learn from it."""
    return not _cut


def log_step(confidence: torch.Tensor, cu_num_logits: torch.Tensor, num_sampled: torch.Tensor,
             estimator=None, idx_mapping: torch.Tensor | None = None) -> None:
    """Append this step's (drafts verified, sampled, confidences) per request to LOG,
    with the estimator's features and coefficients when it has one."""
    global _logged
    if _logged >= _LOG_MAX:
        return
    from vllm.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return
    n = num_sampled.shape[0]
    drafts = (cu_num_logits[1:n + 1] - cu_num_logits[:n] - 1).tolist()
    feats = (estimator.features[idx_mapping].tolist() if estimator is not None and idx_mapping is not None
             else [None] * n)
    rows = [{"k": k, "sampled": s, "conf": [round(c, 4) for c in conf[:k]],
             "feat": None if f is None else [round(x, 3) for x in f[:k]]}
            for k, s, conf, f in zip(drafts, num_sampled.tolist(), confidence.tolist(), feats) if k > 0]
    if rows and estimator is not None and _logged % 200 == 0:
        rows.append({"slope": estimator.slope.tolist(), "intercepts": estimator.intercepts.tolist(),
                     "refits": estimator._refits})
    if rows:
        import json

        with open(LOG, "a") as f:
            f.write(json.dumps(rows) + "\n")
        _logged += 1
