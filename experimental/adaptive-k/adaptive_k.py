# SPDX-License-Identifier: Apache-2.0
"""Choose how many drafts each step verifies from recent draft acceptance.

The drafter proposes num_speculative_tokens drafts every step. Verifying fewer
costs less (each extra verified token adds MoE experts to the step), so when
acceptance falls off early, as it does in free prose, verifying all of them
wastes time. This scheduler picks k, the number of drafts the next step
verifies, from the levels that the dynamic speculative decoding schedule
(num_speculative_tokens_per_batch_size) captured CUDA graphs for.

Per request it keeps c[i], an average of the rate at which draft i is accepted
given that draft i - 1 was. A step that verified d drafts and accepted a < d of
them observes accepts at 0..a-1 and a reject at a. A step that accepted all d
says nothing about positions d and beyond, so those drift towards g, the same
rates averaged over every request, raised or lowered by how far the request's
c[d - 1] is from g[d - 1]. A request that beats the average where it was last
tested is expected to beat it deeper too, so k can climb for it after a run of
poorer requests. g starts each new request's c, and its own unverified
positions drift slowly towards its deepest verified one.

Expected tokens for a step verifying k drafts is 1 + sum over i < k of
c[0] * ... * c[i]. The next k maximises the batch's expected tokens over the
step's cost, and changes only for a gain of at least VLLM_ADAPTIVE_K_MARGIN.

The step's cost comes from a model learned while serving (VLLM_ADAPTIVE_K_ONLINE,
default on): step_ms = base + per_expert * experts(d * t) + per_token * t for t
verified tokens, where experts(n) is how many of the MoE's experts n independent
tokens touch and d the batch's routing diversity (repetitive output touches
fewer experts than its token count suggests). A step's weight reads scale with
the experts it touches, so large steps cost far less than a straight line
through small ones. The three coefficients start from VLLM_ADAPTIVE_K_MODEL,
and a small Kalman filter moves them and d to fit each decode step's measured
time, d much faster than the coefficients. VLLM_ADAPTIVE_K_ONLINE=0 uses the
fixed table VLLM_ADAPTIVE_K_COST_MS ("tokens:ms,...") instead.

With VLLM_ADAPTIVE_K_PER_REQUEST=1, a batch of several requests gets a k per
request instead: every (request, draft) slot is scored by its survival
probability and the best slots are admitted up to the budget that maximises
expected tokens over step cost. Unequal k takes the batch off the full CUDA
graph for attention (piecewise instead).

If the file named by VLLM_ADAPTIVE_K_CONTROL holds "force N", every step
verifies N drafts, which is how the cost table is measured.

VLLM_ADAPTIVE_K_CODE_MODE (Chuck's idea) also keeps rates for two kinds of
output that accept far longer draft runs than prose: code inside ``` fences,
and tool calls between <tool_call> and </tool_call>. Each request keeps them
beside its usual rates, updated only while it is in that mode, and each mode
has its own g. "gated" uses a mode's rates only when every decoding request is
in that mode, since the batch shares one k, and is the default; "1" uses them
per request, and "0" turns code mode off. A fence is a run of three backticks,
which the tokenizer can split across tokens (a closing fence becomes "``" then
"`\n"), so runs are counted across tokens.
"""

import os
import time
from collections import Counter

import numpy as np

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

_ALPHA = float(os.environ.get("VLLM_ADAPTIVE_K_ALPHA", "0.25"))
_MARGIN = float(os.environ.get("VLLM_ADAPTIVE_K_MARGIN", "0.03"))
_PRIOR = 0.8
# g's rate for verified positions, and its drift for unverified ones.
_GLOBAL_ALPHA = 0.05
_GLOBAL_DRIFT = 0.01
_PER_REQUEST = os.environ.get("VLLM_ADAPTIVE_K_PER_REQUEST") == "1"
_CODE_MODE = os.environ.get("VLLM_ADAPTIVE_K_CODE_MODE", "gated")  # "gated", "1" or "0" (off)
_PROSE, _CODE, _TOOL = 0, 1, 2
_ONLINE = os.environ.get("VLLM_ADAPTIVE_K_ONLINE", "1") == "1"
# Fitted on GLM-5.3-Flash at TP=4 on GB10 (k 2, 3, 5 and 7 forced, 1-32 streams, code,
# prose and structured prompts): ms, ms per expert touched, ms per token.
_MODEL = tuple(float(v) for v in (os.environ.get("VLLM_ADAPTIVE_K_MODEL") or "27.0,0.635,0.542").split(","))
# Routed experts and experts per token.
_EXPERTS, _TOPK = 288, 8


class _CostModel:
    """step_ms = base + per_expert * experts(d * t) + per_token * t, learned online.

    An extended Kalman filter over (base, per_expert, per_token, d). Only d
    gets process noise: the workload changes while serving and the hardware
    does not. A run of steps with one size, where only one combination is
    observable, then moves d and leaves the coefficients alone.
    """

    _DRIFT = np.diag([0.0, 0.0, 0.0, 2e-3])

    def __init__(self, theta) -> None:
        self.x = np.array([*theta, 1.0])
        # Confidence in the seed: base to ~5 ms, the slopes to a few percent.
        self.P = np.diag([25.0, 4e-4, 4e-3, 0.1])
        self.n = 0
        self.err = 0.0

    @property
    def d(self) -> float:
        return float(self.x[3])

    @staticmethod
    def experts(n: float) -> float:
        return _EXPERTS * (1.0 - (1.0 - _TOPK / _EXPERTS) ** n)

    def cost(self, tokens: int) -> float:
        base, per_expert, per_token, d = self.x
        return max(base + per_expert * self.experts(d * tokens) + per_token * tokens, 1.0)

    def observe(self, tokens: int, ms: float) -> None:
        predicted = self.cost(tokens)
        self.err += 0.05 * (abs(ms - predicted) / predicted - self.err)
        self.n += 1
        base, per_expert, per_token, d = self.x
        touched = self.experts(d * tokens)
        # d(experts)/dd = (288 - experts) * t * -ln(1 - 8/288)
        slope = (_EXPERTS - touched) * tokens * -np.log1p(-_TOPK / _EXPERTS)
        H = np.array([1.0, touched, float(tokens), per_expert * slope])
        P = self.P + self._DRIFT
        Ph = P @ H
        gain = Ph / (H @ Ph + (0.05 * predicted) ** 2)
        self.x = self.x + gain * (ms - predicted)
        self.x[3] = np.clip(self.x[3], 0.05, 2.0)
        self.P = P - np.outer(gain, Ph)


def _parse_costs(spec: str) -> tuple[np.ndarray, np.ndarray]:
    points = sorted((int(t), float(ms)) for t, ms in (p.split(":") for p in spec.split(",")))
    return np.array([p[0] for p in points], float), np.array([p[1] for p in points], float)


class AdaptiveKScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        lookup = self.dynamic_sd_lookup or [self.num_spec_tokens]
        self._levels = sorted({k for k in lookup[1:] if 0 < k <= self.num_spec_tokens}) or [self.num_spec_tokens]
        self._cost_x, self._cost_y = _parse_costs(
            os.environ.get("VLLM_ADAPTIVE_K_COST_MS", "2:47,4:53,8:65"))
        self._control = os.environ.get("VLLM_ADAPTIVE_K_CONTROL", "")
        self._control_mtime = 0.0
        self._force: int | None = None
        self._rates: dict[str, np.ndarray] = {}
        self._global = np.full(self.num_spec_tokens, _PRIOR)
        # Code mode: per (request, mode) rates, per-mode g, and each request's mode.
        self._mode_rates: dict[tuple[str, int], np.ndarray] = {}
        self._mode_global = {m: np.full(self.num_spec_tokens, _PRIOR) for m in (_CODE, _TOOL)}
        self._mode: dict[str, int] = {}
        self._in_fence: dict[str, bool] = {}
        self._in_tool: dict[str, bool] = {}
        self._scanned: dict[str, int] = {}
        self._run: dict[str, int] = {}  # backticks at the end of each request's output
        self._switches = 0
        self._backticks: dict[int, str] = {}
        self._tool_open = self._tool_close = -1
        if _CODE_MODE != "0":
            self._load_mode_tokens()
        self._k = self.num_spec_tokens
        self._last_log = 0.0
        self._steps: Counter[int] = Counter()
        self._model = _CostModel(_MODEL) if _ONLINE else None
        self._last_output = 0.0
        self._last_model_log = 0.0
        if self._model:
            logger.info("adaptive k: levels %s, online cost model from %s", self._levels, _MODEL)
        else:
            logger.info("adaptive k: levels %s, cost %s", self._levels,
                        dict(zip(self._cost_x.astype(int).tolist(), self._cost_y.tolist())))

    def _load_mode_tokens(self) -> None:
        """The text of every token holding a backtick, and the tool-call markers' ids."""
        from vllm.tokenizers.registry import cached_tokenizer_from_config

        tok = cached_tokenizer_from_config(self.vllm_config.model_config)
        self._backticks = {i: t for i in range(len(tok)) if "`" in (t := tok.decode([i]))}
        self._tool_open = tok.convert_tokens_to_ids("<tool_call>")
        self._tool_close = tok.convert_tokens_to_ids("</tool_call>")
        logger.info("adaptive k: code mode %s, %d tokens hold a backtick, tool call markers %d/%d",
                    _CODE_MODE, len(self._backticks), self._tool_open, self._tool_close)

    def _track_modes(self, req_ids: list[str]) -> None:
        """Advance each request's fence and tool-call state over its new output."""
        for r in req_ids:
            request = self.requests.get(r)
            if request is None:
                continue
            out = request.output_token_ids
            run, fence, tool = self._run.get(r, 0), self._in_fence.get(r, False), self._in_tool.get(r, False)
            for t in out[self._scanned.get(r, 0):]:
                if t == self._tool_open or t == self._tool_close:
                    tool, run = t == self._tool_open, 0
                    continue
                text = self._backticks.get(t)
                if text is None:
                    run = 0
                    continue
                for ch in text:
                    if ch != "`":
                        run = 0
                        continue
                    run += 1
                    fence ^= run == 3
            self._scanned[r], self._run[r], self._in_fence[r], self._in_tool[r] = len(out), run, fence, tool
            mode = _TOOL if tool else _CODE if fence else _PROSE
            self._switches += mode != self._mode.get(r, _PROSE)
            self._mode[r] = mode

    def _active_rates(self, decoding: list[str]) -> list[np.ndarray]:
        """Each request's rates for choosing k: its mode's where code mode says so."""
        if _CODE_MODE == "0":
            return [self._rates[r] for r in decoding]
        modes = [self._mode.get(r, _PROSE) for r in decoding]
        if _CODE_MODE == "gated" and (len(set(modes)) > 1 or modes[0] == _PROSE):
            return [self._rates[r] for r in decoding]
        return [self._rates[r] if m == _PROSE else
                self._mode_rates.setdefault((r, m), self._mode_global[m].copy())
                for r, m in zip(decoding, modes)]

    def update_from_output(self, scheduler_output, model_runner_output):
        if self._model is not None:
            self._time_step(scheduler_output)
        return super().update_from_output(scheduler_output, model_runner_output)

    def _time_step(self, out: SchedulerOutput) -> None:
        """Teach the cost model with this step, if it was a steady decode step.

        Outputs arrive one step apart while the GPU stays busy, so the gap
        between them is the step's time. A step with a prefill chunk is not a
        decode step, a gap far past the prediction includes idle time, and one far
        under it is a timing artifact.
        """
        now = time.monotonic()
        last, self._last_output = self._last_output, now
        tokens = out.total_num_scheduled_tokens
        if not last or not tokens:
            return
        spec = out.scheduled_spec_decode_tokens
        if any(n > 1 + len(spec.get(r, ())) for r, n in out.num_scheduled_tokens.items()):
            return
        ms = (now - last) * 1000.0
        predicted = self._model.cost(tokens)
        if not 0.5 * predicted < ms < 2.0 * predicted + 20.0:
            return
        self._model.observe(tokens, ms)
        if now - self._last_model_log > 30:
            self._last_model_log = now
            m = self._model
            logger.info("adaptive k: cost model base %.1f ms, %.3f ms/expert, %.3f ms/token, diversity %.2f, "
                        "%d steps, mean error %.0f%%, %d code-mode switches", *m.x[:3], m.d, m.n,
                        100 * m.err, self._switches)

    def make_spec_decoding_stats(self, spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                 num_invalid_spec_tokens, request_id):
        if num_draft_tokens:
            self._observe(request_id, num_draft_tokens, num_accepted_tokens)
        return super().make_spec_decoding_stats(spec_decoding_stats, num_draft_tokens, num_accepted_tokens,
                                                num_invalid_spec_tokens, request_id)

    def _observe(self, req_id: str, drafted: int, accepted: int) -> None:
        g = self._global
        c = self._rates.get(req_id)
        if c is None:
            c = self._rates[req_id] = g.copy()
        self._update(c, g, drafted, accepted)
        mode = self._mode.get(req_id, _PROSE)
        if mode != _PROSE:
            gm = self._mode_global[mode]
            self._update(self._mode_rates.setdefault((req_id, mode), gm.copy()), gm, drafted, accepted)

    @staticmethod
    def _update(c: np.ndarray, g: np.ndarray, drafted: int, accepted: int) -> None:
        for rates, alpha in ((c, _ALPHA), (g, _GLOBAL_ALPHA)):
            rates[:accepted] += alpha * (1.0 - rates[:accepted])
            if accepted < drafted:
                rates[accepted] -= alpha * rates[accepted]
        if accepted == drafted:
            lead = c[drafted - 1] - g[drafted - 1]
            c[drafted:] += _ALPHA * (np.clip(g[drafted:] + lead, 0.0, 1.0) - c[drafted:])
            g[drafted:] += _GLOBAL_DRIFT * (g[drafted - 1] - g[drafted:])

    def _step_cost(self, tokens: int) -> float:
        if self._model is not None:
            return self._model.cost(tokens)
        x, y = self._cost_x, self._cost_y
        if tokens <= x[-1]:
            return float(np.interp(tokens, x, y))
        return float(y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (tokens - x[-1]))

    def _read_control(self) -> None:
        if not self._control:
            return
        try:
            mtime = os.stat(self._control).st_mtime
        except FileNotFoundError:
            self._force = None
            return
        if mtime != self._control_mtime:
            self._control_mtime = mtime
            words = open(self._control).read().split()
            self._force = int(words[1]) if len(words) == 2 and words[0] == "force" else None
            logger.info("adaptive k: control file says %s", self._force or "adapt")

    def _choose(self, req_ids: list[str]) -> int:
        self._read_control()
        if self._force is not None:
            return min(max(self._force, 1), self.num_spec_tokens)
        decoding = [r for r in req_ids if r in self._rates]
        if not decoding:
            return self._k
        survival = np.cumprod(np.stack(self._active_rates(decoding)), axis=1).sum(axis=0)
        expected = {k: len(decoding) + survival[:k].sum() for k in self._levels}
        score = {k: expected[k] / self._step_cost(len(decoding) * (k + 1)) for k in self._levels}
        best = max(score, key=score.get)
        current = self._k if self._k in score else best
        return best if score[best] > score[current] * (1 + _MARGIN) else current

    def _choose_per_request(self, req_ids: list[str]) -> dict[str, int]:
        """A k for each request from one draft budget: every (request, draft)
        slot is scored by its survival probability, and the budget is the
        number of best slots that maximises expected tokens over step cost."""
        decoding = [r for r in req_ids if r in self._rates]
        if not decoding:
            return {}
        survival = np.cumprod(np.stack([self._rates[r] for r in decoding]), axis=1)
        order = np.argsort(-survival, axis=None, kind="stable")
        best_b, best_score, n = 0, len(decoding) / self._step_cost(len(decoding)), len(decoding)
        gained = 0.0
        for b, flat in enumerate(order, 1):
            gained += survival.flat[flat]
            score = (n + gained) / self._step_cost(n + b)
            if score > best_score:
                best_b, best_score = b, score
        counts = np.bincount(order[:best_b] // survival.shape[1], minlength=n)
        return {r: max(int(c), 1) for r, c in zip(decoding, counts)}

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in [r for r in self._rates if r not in self.requests]:
            del self._rates[req_id]
            for d in (self._mode, self._in_fence, self._in_tool, self._scanned, self._run):
                d.pop(req_id, None)
            for m in (_CODE, _TOOL):
                self._mode_rates.pop((req_id, m), None)
        if _CODE_MODE != "0":
            self._track_modes(list(scheduler_output.num_scheduled_tokens))
        self._read_control()
        if self.num_spec_tokens and _PER_REQUEST and self._force is None:
            per = self._choose_per_request(list(scheduler_output.num_scheduled_tokens))
            if len(per) > 1:
                scheduler_output.num_spec_tokens_to_schedule = max(per.values())
                super()._update_after_schedule(scheduler_output)
                for req_id, k in per.items():
                    request = self.requests.get(req_id)
                    if request is not None and request.spec_token_ids:
                        request.spec_token_ids = [-1] * k
                for k in per.values():
                    self._steps[k] += 1
                return
        self._update_batch_k(scheduler_output)

    def _update_batch_k(self, scheduler_output: SchedulerOutput) -> None:
        if self.num_spec_tokens:
            k = self._choose(list(scheduler_output.num_scheduled_tokens))
            self._steps[k] += 1
            now = time.monotonic()
            if k != self._k and now - self._last_log > 5:
                self._last_log = now
                logger.info("adaptive k: %d -> %d; steps per k so far %s", self._k, k, dict(sorted(self._steps.items())))
            self._k = k
            scheduler_output.num_spec_tokens_to_schedule = self._k
        super()._update_after_schedule(scheduler_output)
