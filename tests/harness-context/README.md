# Harness context exercise

`fixture.py` is the exact 100-round helper used for the recorded DeepSeek Harness run. `recorded-audit-20261009.jsonl` is its original generated-data audit: 100 inventories and 100 accepted submissions. Both were checked against the supplied session export and independent exhaustive scoring. The audit contains generated item IDs, integer budgets/costs/values, submitted/expected plans and acceptance flags. Private session prompts, reasoning transcripts and connection settings are excluded.

See [results and limits](../../docs/TOOL_REASONING_VALIDATION.md#deepseek-harness-run) and [machine-readable counts and hashes](../../manifests/harness-tool-reasoning-20261009.json).

## Run another session

Attach `fixture.py` to a clean tool-enabled Harness session, select the GLM deployment and enable thinking. Give it this prompt:

```text
Run the attached fixture.py exercise in this single conversation.
Create a NEW isolated folder in the writable workspace and copy fixture.py
there unchanged. Only modify files in that folder. Do not install packages,
delegate to another agent, access credentials or change any service.

Run python3 fixture.py start, then complete rounds 0 through 99 in order.
For each CURRENT_INVENTORY, choose exactly three distinct items with total
cost at most the current budget. A and C are incompatible. E requires F.
Include at least one of B or G. Maximize total value; break ties using the
alphabetically smallest sorted IDs. Calculate the proposal yourself.

Submit one proposal per separate tool call, using this command shape:
python3 fixture.py submit '{"round":N,"ids":["X","Y","Z"],"cost":C,"value":V}'
Replace all placeholders with the current round's actual proposal. Use the
isolated folder as the tool's working directory. Wait for each tool result
and inspect the new inventory before deciding the next proposal. Do not
batch rounds, pre-generate inventories, invoke another solver or read/edit
state.json or audit.jsonl. The helper validates proposals and logs evidence.

The numeric archive is already processed background; take no action on it
and do not echo it. Keep ordinary progress messages short and use the normal
reasoning channel. Do not voluntarily compact or replace conversation history.
Every archive-bearing result must include BEGIN_PAYLOAD, CURRENT_INVENTORY
and END_PAYLOAD. Stop and report truncation, compaction or tool errors.

If rejected, reassess that inventory and retry at most twice, then stop if
still rejected. Stop after EXERCISE_COMPLETE. If reliable per-request input
usage exceeds 900,000, stop after the current round to leave output headroom.
Do not infer current context size by summing cumulative usage.

Report accepted/rejected rounds, completion status, any truncation/compaction,
the largest actually exposed input-token count (or unknown), and the folder.
Leave the generated fixture.py, audit.jsonl and state.json for inspection.
```

The helper writes state and an audit beside itself, so use the isolated copy rather than running it in this tracked directory. Each new inventory is random; reruns follow the same rules but will produce different values and plans. The published audit is historical evidence, not an input for generating a new run. The helper is not part of `verify.sh` and does not contact the model API itself; the Harness performs the model/tool loop.

At the tested tokenizer, one approximately 9.9 KB result added about 6,005 tokens. One hundred retained results therefore aim for roughly 600k input tokens before conversation history. Actual context depends on the client, reasoning output and truncation/compaction; inspect per-request usage in the exported logs. This creates sustained inference load and can take about an hour. It does not establish latency benchmarks or arbitrary long-document retrieval quality.
