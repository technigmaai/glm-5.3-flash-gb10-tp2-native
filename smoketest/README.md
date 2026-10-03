# smoketest

    smoketest/run.sh http://<head>:8002          # the head's API
    smoketest/run.sh http://<head>:6381          # through mentatd-serve

Cases are shell functions in [run.sh](run.sh); checks are jq expressions. Needs
curl and jq. The served name defaults to `name:` in `model.yaml`; pass a second
argument to override it. Exit status is the number of failed cases, and
`ONLY=t_fact smoketest/run.sh ...` runs one case. It passed 8/8 through
mentatd-serve on 2026-09-23.

`page-table.png` is a synthetic page with one question that can only be
answered by reading it.

Speed, recall and determinism are not here: they take minutes, and belong in a
benchmark on a quiet box.
