# Functional smoke tests

Run the complete deployment verification from the head checkout with:

```bash
./verify.sh
```

This checks configured aliases, the eight functional cases below and concurrent
short streams at the configured sequence limit. It submits real inference
requests; run it after the API and startup self-test are ready.

To run only the functional cases, use the head API on port 8000 and pass a served
model alias explicitly:

```bash
bash tests/smoketest/run.sh http://127.0.0.1:8000 glm53
# From another machine, replace 127.0.0.1 with the head address.
ONLY=t_fact bash tests/smoketest/run.sh http://127.0.0.1:8000 glm53
```

The alias argument is required for this native recipe; it has no model.yaml.
There is one native container on each node and no Mentat proxy endpoint.
The host needs curl, jq and Bash. Exit status is the number of failed cases.

Cases in [run.sh](run.sh) check the checkpoint identity, default/low-effort
thinking, a known fact, parsed tool calls, image reading, an explicit client token
limit and rejection of a request exceeding the configured context limit.
`page-table.png` is a synthetic page whose answer requires reading the image.

All eight cases passed on the native deployment on 2026-10-07. The checks use
finite output limits and do not qualify million-token requests, maximum image
loads, speed, long-term stability or determinism.
