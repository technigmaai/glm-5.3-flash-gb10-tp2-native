# External feedback supplied on 2026-10-04

This summarizes a tester's report supplied by the maintainer. It is attributed
external feedback, not a benchmark run or independently reproduced result from
this deployment's local validation. No public source URL was supplied.

The tester described the native TP2 C6 recipe at commit `24ba8fb`, the published
`c748079-displaykv1-arm64-cu130` image, the pinned NVIDIA model and DFlash2,
6,144 batched tokens, six slots and 1,047,552 configured context tokens.
They used **7.2 GiB KV per GPU** (`7730941132` bytes) and reported 1,215,513
KV tokens. Their 64-point cold sweep covered 4,096–262,144 prompt tokens,
with unique seeded prompts and 256 generated tokens. They also reported a
300k sanity check and six tool-eval runs (three high and three max effort).

| Reported measurement | Result |
|---|---:|
| Average cold prefill | 2,678 tokens/s |
| Average generation in the sweep | 42.7 tokens/s |
| TTFT at 262k | 100.4 s |
| Tool-eval score, high / max | 92.0 / 96.0 |
| Tool-eval utility, high / max | 278 / 205 points per hour |
| Time per tool-eval run, high / max | 19.8 / 28.1 min |

The tester reported engine failures around 155k prompt tokens with a 9 GiB
pool on their machines, and described their 7.2 GiB runs as stable. Their
failure diagnosis, precision description and comparative performance claims
have not been independently verified here.

**Our local observation:** we have not seen OOM during the recorded 9 GiB
startup, basic API/image/concurrency checks or image publication. These checks
do not prove that OOM is impossible under a different or heavier workload.
We chose **8 GiB** for additional RAM headroom and retain **9 GiB as an option**.
The external 7.2 GiB measurements do not qualify our 8 GiB profile or a
million-token workload. Exact results depend on the complete workload and
configuration; the figures above should not be presented as our measurements.
