# Tool-continuation reasoning validation — 2026-10-09

The template reminder restored separate thinking in the reproduced failing tool continuation. The original template produced zero reasoning tokens at approximately 38k, 500k and 900k input tokens, and raw completions began with `</think>`. With the reminder, those same cases returned separate reasoning and correct plans. The reminder is conditional on thinking being enabled and the final message being a tool result; `chat_template_kwargs.tool_reasoning_reminder=false` opts out.

This is a prompt/template mitigation for the reproduced behavior. It does not repair missing client history, reclassify ordinary content as reasoning, or guarantee that the model always emits thinking after every tool call. Clients must preserve the complete reasoning/tool history.

## Fresh-data follow-up

Three short rounds generated genuine assistant reasoning and tool calls. Accepted submissions then revealed new inventory, prices, values and budgets in tool results. Future inventories were not included in earlier messages. An exhaustive scorer checked all 56 three-item combinations against budget, incompatibility, prerequisite, required-item and alphabetical tie-break constraints.

The complete short history was expanded with synthetic processed archive records. At each size, paired requests differed only in the reminder flag; the reminder-on response then fed two further tool continuations. Instructions and newly revealed data were near the end of the input. The same short seed was retained for the subsequent 900k/1.02M run.

The 600k follow-up passed all seven requests: three seed rounds, three long tool continuations and one reminder-off control. The larger follow-up passed all eight requests:

| Case | Input tokens | Separate reasoning tokens | Correct optimal plan |
|---|---:|---:|---|
| 900k batch 3, reminder off | 900,008 | 1,174 | Yes |
| 900k batch 3, reminder on | 900,046 | 1,278 | Yes |
| 900k batch 4, reminder on | 901,793 | 1,299 | Yes |
| 900k batch 5, reminder on | 903,575 | 1,355 | Yes |
| 1.02M batch 3, reminder on | 1,019,983 | 979 | Yes |
| 1.02M batch 3, reminder off | 1,019,945 | 1,112 | Yes |
| 1.02M batch 4, reminder on | 1,021,463 | 1,039 | Yes |
| 1.02M batch 5, reminder on | 1,023,032 | 1,256 | Yes |

All responses finished with parsed tool calls. No raw think/tool markers leaked into ordinary content. Tokenizer counts matched API prompt usage; a separate local scorer verified plans and reconstructed reasoning/content from the recorded streams. Sanitized counts and generated plans are in [the machine-readable results](../manifests/tool-reasoning-checks-20261009.json). Private session exports and server logs are not included.

**The reminder-off controls at 600k, 900k and 1.02M also passed.** These fresh fixtures did not reproduce missing thinking or establish that the reminder was necessary for them. The earlier failing-session replay provides the before/after evidence for the mitigation. Synthetic padding does not qualify arbitrary million-token retrieval, long production conversations, all community reports, or general model accuracy.

The existing 8 GiB KV / C6 / 6,144 batched-token / 1,047,552 context profile was unchanged. No OOMs, preemptions or unexpected container restarts were observed during the follow-ups. Both containers remained running; final API health was HTTP 200. Six slots are not six simultaneous million-token contexts.

## DeepSeek Harness run

A subsequent clean DeepSeek Harness session completed all 100 fresh planning
rounds. Its largest recorded per-request input was **774,863 tokens**. Separate
reasoning was present in **104/104 responses**, including **98/98 responses
immediately following a tool result**. All **100 plans** were independently
recomputed as optimal; no proposal was rejected.

| Input threshold | Responses | Direct tool continuations with separate reasoning |
|---|---:|---:|
| 450k | 43 | 40/40 |
| 500k | 37 | 35/35 |
| 600k | 23 | 21/21 |
| 700k | 10 | 9/9 |

All 100 tool payloads included their boundary markers, current inventory and
128 exact numeric archive records. No tool errors, truncation or recorded
compaction were found; reported input counts rose monotonically. Reasoning
stream text exactly matched stored reasoning blocks in all 104 responses, and
no raw think tags appeared in ordinary assistant content. The run took about
55 minutes with reasoning effort `max`; its client output allowance was 65,536
tokens, separate from the server context limit.

The separately supplied audit contained 100 inventories and 100 accepted
submissions. Every inventory and proposal matched its session tool output/call,
and every expected/actual plan matched independent exhaustive scoring. The
[original fixture](../tests/harness-context/fixture.py) was byte-identical to the
supplied exercise. The [reviewed audit](../tests/harness-context/recorded-audit-20261009.jsonl)
contains only generated test data. [Machine-readable results and hashes](../manifests/harness-tool-reasoning-20261009.json)
and [Harness reproduction instructions](../tests/harness-context/README.md) are included.
The raw session export and final state file are excluded.

The Harness injected five periodic time-context messages after the initial
step; those replies are excluded from the direct-tool counts above. Usage and
reasoning channels come from the processed Harness export; raw outgoing HTTP
payloads were not available. There was no reminder-off control in this session,
so it does not establish that the reminder was necessary for this particular
fixture. This is successful end-to-end evidence through 774k for this Harness,
with synthetic background and active tasks near the end. It does not qualify
all harnesses, arbitrary long-document retrieval or every community report.

## Reproduce the fresh-data procedure

The [standard-library API driver](../tests/verify-long-tool-reasoning.py) generates its own fixtures and genuine seed responses. It submits real, sequential inference requests and saves generated reasoning, content, tool calls and SSE records to a new output directory. It changes no deployment files and executes no model-generated terminal commands. It is opt-in and is not part of `verify.sh`.

Run where the head API is reachable, using an API root URL without `/v1`:

```bash
python3 tests/verify-long-tool-reasoning.py \
  --base-url http://YOUR_HEAD_IP:8000 \
  --model glm53 \
  --targets 600000 900000 1020000 \
  --output-dir /tmp/glm-tool-reasoning-check
```

Choose targets that fit the configured context limit, leaving room for outputs and later turns. The recorded 1.02M target fits the 1,047,552-token profile. Large prefills can take many minutes and occupy the serving model; run when that load is appropriate. Use a new/empty output directory for each run. If the endpoint requires authentication, the script reads `OPENAI_API_KEY` without writing it into results.

Sampling is temperature 0, top_p 1, seed 739, `reasoning_effort=max`, `enable_thinking=true`, `clear_thinking=false`, and an 8,192-token output allowance per test request. This allowance is not a server-wide reply cap. Generated seed outputs can vary across software/hardware versions; reproduction means repeating the procedure, not obtaining byte-identical transcripts. Pair order reverses at alternate sizes, and shared history may be cached; these runs are not latency benchmarks.

A reminder-on failure is saved and stops the run with a nonzero exit. Reminder-off failures are retained as control observations and are never used as subsequent assistant history. The driver checks optimal-plan correctness, separate reasoning, prompt-token accounting and tool-call completion.
