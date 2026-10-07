#!/usr/bin/env bash
set -euo pipefail
DEPLOY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$DEPLOY"
[[ -f .env ]] || { echo 'Copy .env.example to .env and configure this node first.' >&2; exit 1; }
set -a; source .env; set +a
PROJECT_NAME=${PROJECT_NAME:-glm53-native}
compose() { docker compose --env-file "$DEPLOY/.env" -p "$PROJECT_NAME" -f "$DEPLOY/compose.yaml" "$@"; }
head_only() { [[ ${ROLE:?} == head ]] || { echo 'Run this on the configured head node.' >&2; exit 1; }; }
peer() { ssh -o BatchMode=yes -o ConnectTimeout=10 "${PEER_SSH:?Set PEER_SSH}" "$@"; }
peer_cluster() {
  local remote
  printf -v remote '%q ' bash "${PEER_DEPLOY_DIR:?Set PEER_DEPLOY_DIR}/scripts/cluster.sh" "$@"
  peer "$remote"
}
approved() { [[ ${1:-} == --approved ]] || { echo 'This changes containers. Supply --approved for the requested operation.' >&2; exit 2; }; }
lock() { exec 9>"${CLUSTER_LOCK:-${LOG_HOST_DIR:?Set LOG_HOST_DIR}/cluster.lock}"; flock -n 9 || { echo 'A cluster operation is already in progress.' >&2; exit 1; }; }
start_nodes() {
  python3 "$DEPLOY/scripts/check.py"; peer_cluster check
  python3 "$DEPLOY/scripts/preflight.py"; peer_cluster node-preflight
  compose up -d; peer_cluster node-up --approved
  echo "Cluster starting. API: http://${HEAD_HOST}:${API_PORT:-8000}/v1"
}
cmd=${1:-status}; shift || true
case "$cmd" in
  config) compose config "$@" ;;
  check) python3 "$DEPLOY/scripts/check.py" ;;
  node-preflight) python3 "$DEPLOY/scripts/preflight.py" ;;
  node-status) compose ps --all ;;
  status)
    echo "Local ${ROLE} rank:"; compose ps --all
    if [[ "$ROLE" == head ]]; then
      echo 'Peer worker rank:'; peer_cluster node-status
      if curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:${API_PORT:-8000}/health" >/dev/null; then
        echo 'Head API: healthy'
        curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:${API_PORT:-8000}/v1/models" | python3 -c 'import json,sys; print("Model aliases: " + ", ".join(m["id"] for m in json.load(sys.stdin)["data"]))'
      else echo 'Head API: unavailable (loading or failed); inspect both node logs.'; fi
    fi
    ;;
  node-up) approved "${1:-}"; compose up -d ;;
  node-down) approved "${1:-}"; compose down --timeout 60 ;;
  start) head_only; approved "${1:-}"; lock; start_nodes ;;
  stop) head_only; approved "${1:-}"; lock; compose down --timeout 60; peer_cluster node-down --approved ;;
  restart)
    head_only; approved "${1:-}"; lock
    python3 "$DEPLOY/scripts/check.py"; peer_cluster check
    compose down --timeout 60; peer_cluster node-down --approved; sleep 5; start_nodes
    ;;
  cutover)
    head_only; approved "${1:-}"; lock
    python3 "$DEPLOY/scripts/check.py"; peer_cluster check
    "$DEPLOY/scripts/legacy.sh" stop --approved; sleep 5; start_nodes
    ;;
  rollback)
    head_only; approved "${1:-}"; lock
    : "${LEGACY_REPO:?No legacy deployment configured}"
    compose down --timeout 60; peer_cluster node-down --approved; flock -u 9
    "$DEPLOY/scripts/legacy.sh" start --approved
    ;;
  sync) head_only; python3 "$DEPLOY/scripts/sync-repo.py" "$@" ;;
  verify)
    head_only
    export VERIFY_BASE_URL="http://127.0.0.1:${API_PORT:-8000}"
    export VERIFY_METRICS_URL="$VERIFY_BASE_URL/metrics"
    export VERIFY_MODEL_ALIASES="${MODEL_ALIASES:-${SERVED_NAME:-glm53}}"
    export VERIFY_CONCURRENCY="${MAX_NUM_SEQS:-4}"
    python3 "$DEPLOY/tests/verify-api.py"
    bash "$DEPLOY/tests/smoketest/run.sh" "$VERIFY_BASE_URL" "${SERVED_NAME:-glm53}"
    python3 "$DEPLOY/tests/verify-concurrency.py"
    ;;
  node-logs) compose logs "$@" ;;
  logs)
    node=${1:-head}
    if [[ "$node" == head || "$node" == worker ]]; then shift || true; else node=head; fi
    if [[ "$node" == worker && "$ROLE" == head ]]; then peer_cluster node-logs "$@"; else compose logs "$@"; fi
    ;;
  *) echo 'Use root start.sh/stop.sh/restart.sh/status.sh/tail-log.sh/check.sh/sync-repo.sh/verify.sh. Mutations require --approved.' >&2; exit 2 ;;
esac
