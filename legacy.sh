#!/usr/bin/env bash
# Optional adapter for a previous Kindling/Mentat installation.
set -euo pipefail
DEPLOY=$(dirname "$(readlink -f "$0")")
set -a; source "$DEPLOY/.env"; set +a
[[ ${2:-} == --approved ]] || { echo 'Legacy operations require --approved.' >&2; exit 2; }
: "${LEGACY_REPO:?Set LEGACY_REPO}" "${LEGACY_LAUNCHER:?Set LEGACY_LAUNCHER}"
case "${1:-}" in
  stop)
    "$LEGACY_REPO/$LEGACY_LAUNCHER" stop
    config=${LEGACY_CONFIG_DIR:?Set LEGACY_CONFIG_DIR}
    docker compose --env-file "$LEGACY_REPO/$config/mentat.env" -p glm53-mentat -f "$LEGACY_REPO/$config/mentatd.yaml" down --timeout 60
    remote="cd $(printf '%q' "$LEGACY_REPO") && docker compose --env-file $(printf '%q' "$config/mentat.env") -p glm53-mentat -f $(printf '%q' "$config/mentatd.yaml") down --timeout 60"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${PEER_SSH:?}" "$remote"
    ;;
  start) "$LEGACY_REPO/$LEGACY_LAUNCHER" start ;;
  *) echo 'Usage: legacy.sh {stop|start} --approved' >&2; exit 2 ;;
esac
