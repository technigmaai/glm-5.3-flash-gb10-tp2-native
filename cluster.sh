#!/usr/bin/env bash
set -euo pipefail
DEPLOY=$(dirname "$(readlink -f "$0")")
cd "$DEPLOY"
[[ -f .env ]] || { echo 'Create .env from .env.example or run configure.py first.' >&2; exit 1; }
set -a; source .env; set +a
PROJECT_NAME=${PROJECT_NAME:-glm53-native}
compose() { docker compose --env-file "$DEPLOY/.env" -p "$PROJECT_NAME" -f "$DEPLOY/compose.json" "$@"; }
head_only() { [[ ${ROLE:?} == head ]] || { echo 'Run this on the configured head node.' >&2; exit 1; }; }
peer() { ssh -o BatchMode=yes -o ConnectTimeout=10 "${PEER_SSH:?Set PEER_SSH}" "$@"; }
peer_cluster() {
  local remote
  printf -v remote '%q ' bash "${PEER_DEPLOY_DIR:?Set PEER_DEPLOY_DIR}/cluster.sh" "$@"
  peer "$remote"
}
approved() { [[ ${1:-} == --approved ]] || { echo 'This changes running containers. Supply --approved only after authorization.' >&2; exit 2; }; }
lock() { exec 9>"${CLUSTER_LOCK:-$DEPLOY/.cluster.lock}"; flock -n 9 || { echo 'A cluster operation is already in progress.' >&2; exit 1; }; }
start_nodes() {
  python3 "$DEPLOY/check.py"; peer_cluster check
  python3 "$DEPLOY/preflight.py"; peer_cluster node-preflight
  compose up -d; peer_cluster node-up --approved
  echo "Cluster starting. API: http://${HEAD_HOST}:${API_PORT:-8000}/v1"
}
cmd=${1:-status}; shift || true
case "$cmd" in
  config) compose config "$@" ;;
  check) python3 "$DEPLOY/check.py" ;;
  node-preflight) python3 "$DEPLOY/preflight.py" ;;
  status) compose ps; [[ "$ROLE" != head ]] || peer_cluster status ;;
  node-up) approved "${1:-}"; compose up -d ;;
  node-down) approved "${1:-}"; compose down --timeout 60 ;;
  start) head_only; approved "${1:-}"; lock; start_nodes ;;
  stop) head_only; approved "${1:-}"; lock; compose down --timeout 60; peer_cluster node-down --approved ;;
  restart)
    head_only; approved "${1:-}"; lock
    python3 "$DEPLOY/check.py"; peer_cluster check
    compose down --timeout 60; peer_cluster node-down --approved; sleep 5; start_nodes
    ;;
  cutover)
    head_only; approved "${1:-}"; lock
    python3 "$DEPLOY/check.py"; peer_cluster check
    "$DEPLOY/legacy.sh" stop --approved; sleep 5; start_nodes
    ;;
  rollback)
    head_only; approved "${1:-}"; lock
    : "${LEGACY_REPO:?No legacy deployment configured}"
    compose down --timeout 60; peer_cluster node-down --approved; flock -u 9
    "$DEPLOY/legacy.sh" start --approved
    ;;
  verify)
    head_only
    export VERIFY_BASE_URL="http://127.0.0.1:${API_PORT:-8000}"
    export VERIFY_METRICS_URL="$VERIFY_BASE_URL/metrics"
    export VERIFY_MODEL_ALIASES="${MODEL_ALIASES:-${SERVED_NAME:-glm53}}"
    export VERIFY_CONCURRENCY="${MAX_NUM_SEQS:-4}"
    python3 "$DEPLOY/verify-api.py"
    bash "$DEPLOY/smoketest/run.sh" "$VERIFY_BASE_URL" "${SERVED_NAME:-glm53}"
    python3 "$DEPLOY/verify-concurrency.py"
    ;;
  logs) compose logs "$@" ;;
  *) echo 'Usage: cluster.sh {check|config|status|start|stop|restart|cutover|rollback|verify|logs}; mutations require --approved' >&2; exit 2 ;;
esac
