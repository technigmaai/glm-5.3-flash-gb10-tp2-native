#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if (( $# == 0 )); then set -- --follow; fi
exec bash "$ROOT/scripts/cluster.sh" logs "$@"
