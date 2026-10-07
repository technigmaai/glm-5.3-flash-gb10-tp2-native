#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Serve GLM-5.3-Flash (NVFP4) across the ConnectX-linked GB10 nodes. TP and PP
# come from the environment, so the same image serves the TP=2/PP=2 pair-of-
# pairs and a flat TP=4 across one fabric.
#
# Deliberately simpler than ds4-flash/entrypoint.sh in one respect: it loads
# PLAIN SAFETENSORS and builds no sharded-state cache. That costs boot time and
# buys correctness margin -- the sharded-state path needs the modelopt
# IDEMPOTENT-GUARD patch, because a cache is dumped AFTER the NVFP4
# kernel-format transform and re-running that transform on load permutes the
# fused gate/up halves and serves fluent nonsense with no error. This image is
# built on upstream's per-model base and carries none of our patches, so the
# safe load is the one that runs the transform exactly once.
#
# Adopt the sharded-state cache only together with that patch. See
# ds4-flash/vllm-spark.patch.
# ---------------------------------------------------------------------------

TP="${TP:-4}"
MTP="${MTP:-1}"
FABRIC_LAYOUT=mesh

# --- per-TP defaults ---------------------------------------------------------
# Each rank holds 1/TP of the weights and of every request's KDA state, so
# the memory left for KV, and how many requests fit in it, depend on TP. These
# fill in whatever .env leaves out. TP=4 is the measured four-box setup. TP=2
# is two boxes with 89.6 GiB of weights per rank. Halving the batched-token
# budget to 8192 shrinks the activation peak by ~2.8 GiB on the head, at
# ~9% of prefill speed, and an 8 GiB KV pin then leaves the head 1.4 GiB free
# at its lowest. At 16384 an 8 GiB pin got a worker OOM-killed mid-prefill.
# Each rank carries half the KDA heads, so a request's states are twice as big.
# RecoverSSM (recoverssm.yaml) keeps one state per request instead of 1 + k:
# 64 run at TP=4, the most whose k=7 steps fit the largest captured graph,
# and 16 at TP=2. It also removes the state
# writes adaptive-k's step cost used to include, so it gets its own prior,
# fitted on code and prose at two concurrencies with k forced to 2 and 7.
_rs=0; [[ "${VLLM_GLM5NEXT_RECOVERSSM:-0}" == 1 ]] && _rs=1
case "$TP" in
  4) # KV_EXTRA_MEMORY (bytes) raises the default pin. dispram.yaml sets 2 GiB,
     # the RAM kindling-spark-os's 64 KiB kernel returns.
     : "${KV_CACHE_MEMORY:=$(( 27917287424 + ${KV_EXTRA_MEMORY:-0} ))}" "${MAX_MODEL_LEN:=524288}"
     (( _rs )) && : "${MAX_NUM_SEQS:=64}" "${VLLM_ADAPTIVE_K_MODEL:=26.4,0.660,0.484}"
     # With the drafter's KV in its own pool (fixes.yaml), ~50 requests' KDA
     # states fit in the pool; without it, 32.
     [[ "${VLLM_GLM5NEXT_DRAFT_POOL:-0}" == 1 ]] && : "${MAX_NUM_SEQS:=50}"
     : "${VLLM_ADAPTIVE_K_MODEL:=27.0,0.635,0.542}" ;;
  3) # Three boxes on the zero-padded checkpoint (experimental/tp3). 26 GiB of
     # KV had the host OOM-killer take rank 0 during graph capture; 12 GiB
     # holds 1.4M tokens (1.7M with RecoverSSM). Block 3456 becomes 3584 in
     # vLLM: 14% mamba page padding, where the 4608 it picks from 2304 pads 46%
     # and coarsens the prefix cache.
     : "${KV_CACHE_MEMORY:=12884901888}" "${MAX_MODEL_LEN:=524288}" "${BLOCK_SIZE:=3456}"
     export BLOCK_SIZE
     # With RecoverSSM, 64 requests at k=7 fill the 512-token graph cap as at
     # TP=4: +24% aggregate at 48 streams over 32, single-stream unchanged.
     (( _rs )) && : "${MAX_NUM_SEQS:=64}"
     # What the online fit converged to at TP=3 with RecoverSSM on (3% mean
     # error; the two three-box sets agreed within 2%).
     : "${VLLM_ADAPTIVE_K_MODEL:=27.5,0.795,0.50}" ;;
  6) # Six boxes on the same zero-padded checkpoint as TP=3 (66 / 66 / 2304
     # split as 11 heads and 384 MoE columns per rank). Weights are ~1/6 per
     # rank, so the TP=4 KV pin fits. With 11 KDA heads a rank's mamba page is
     # ~1573 tokens of attention page: block 2304 pads it 46%, 1792 (the
     # smallest multiple of 256 above it) 14%, for a 1.7% bigger pool and
     # 17% faster cached TTFT.
     : "${KV_CACHE_MEMORY:=27917287424}" "${MAX_MODEL_LEN:=524288}" "${BLOCK_SIZE:=1792}"
     export BLOCK_SIZE
     (( _rs )) && : "${MAX_NUM_SEQS:=64}"
     # What the online fit converged to at TP=6 with RecoverSSM on (1% error).
     : "${VLLM_ADAPTIVE_K_MODEL:=19.3,0.50,0.40}" ;;
  2) : "${KV_CACHE_MEMORY:=8589934592}" "${MAX_MODEL_LEN:=163840}" "${MAX_NUM_BATCHED_TOKENS:=8192}"
     # A rank reads twice the expert weights it does at TP=4, so the
     # per-expert cost is held at twice TP=4's and the rest fitted.
     (( _rs )) && : "${MAX_NUM_SEQS:=16}" "${VLLM_ADAPTIVE_K_MODEL:=32.1,1.32,0.427}"
     : "${MAX_NUM_SEQS:=4}"
     : "${VLLM_ADAPTIVE_K_MODEL:=35.0,1.27,0.8}" ;;
  *) echo "FATAL: TP=$TP; this recipe is tuned for TP=4 or TP=RING4 (four boxes), TP=6 (six), TP=3 (three) or TP=2 (two)" >&2; exit 1 ;;
esac
export VLLM_ADAPTIVE_K_MODEL
# SPEC_METHOD picks the drafter: dflash (the separate DFlash2 draft model at
# DFLASH_MODEL, which must be mounted on every node; k must be 7, its block
# size minus one, and the model refuses anything else), mtp (the checkpoint's
# own head, num_nextn_predict_layers: 1), or none. MTP=0 is the older off
# switch and still turns speculation off whichever method is named. Resolved
# here, above the cache tag, because the tag hashes it.
#
# dflash is what serves: 109.8 / 88.8 / 52.6 tok/s on structured / code / prose
# with v6, where MTP gave 57.2 / 54.4 / 45.6 with v4 (both 2026-09-23). Judge a drafter
# on acceptance LENGTH, not acceptance rate. Set here rather than through
# EXTRA_ARGS, which is one string that a compose override restating it
# silently replaces.
SPEC_METHOD="${SPEC_METHOD:-dflash}"
[[ "$MTP" == "0" ]] && SPEC_METHOD=none
# Fixed native ranks: no daemon, discovery, election, or Ray agent.
: "${NODE_RANK:?set NODE_RANK}" "${HEAD_HOST:?set HEAD_HOST}" "${VLLM_HOST_IP:?set VLLM_HOST_IP}"
case "$NODE_RANK:$ROLE" in 0:head|1:worker) ;; *) echo 'FATAL: native rank/role mismatch' >&2; exit 1;; esac
[[ "$TP" == 2 ]] || { echo 'FATAL: this native profile requires TP=2' >&2; exit 1; }
if [[ "$ROLE" == head && "$VLLM_HOST_IP" != "$HEAD_HOST" ]] ||
   [[ "$ROLE" == worker && "$VLLM_HOST_IP" == "$HEAD_HOST" ]]; then
  echo 'FATAL: native role/address mismatch' >&2; exit 1
fi
NATIVE_DRY_RUN="${GLM53_NATIVE_DRY_RUN:-0}"

MODEL="${MODEL_DIR:-/models/glm-5.3-flash-nvfp4}"
SERVED="${SERVED_NAME:-glm53}"

# --- cluster networking: everything rides the ConnectX link -----------------
# No literal address appears here or in the compose file. The boxes are NOT
# symmetric (one carries its link on enp1s0f1np1, the rest on enp1s0f0np0), so
# any hardcoded interface name is wrong somewhere whichever you pick.
#
# VLLM_HOST_IP is this node's LAN address: mentat identifies a node by it, and
# the agent and daemon must agree on one string. Address prefixes
# (CLUSTER_SUBNET, FABRIC_SUBNETS) name the fabric. Whatever .env leaves out
# comes from the local mentatd: its lan-tagged address, and one prefix per
# rdma-tagged address.
#
# An uncabled port powers down completely -- no PCI device, no
# /sys/class/infiniband entry. That is not a missing driver and no amount of
# modprobe fixes it, so a node reports only the ports it actually has.
: "${FABRIC_SUBNETS:?set FABRIC_SUBNETS}"
CLUSTER_SUBNET="${CLUSTER_SUBNET:-${FABRIC_SUBNETS%% *}}"
export VLLM_HOST_IP

# Gloo needs an EXACT interface name -- NCCL_SOCKET_IFNAME takes a prefix, this
# does not. Left unset, Gloo binds 127.0.0.1 and the cluster fails silently at
# rendezvous, so derive it from whichever interface actually owns VLLM_HOST_IP.
# Bounded, unlike the fabric wait above: the address is already up, so the
# interface and its RDMA device are local state that either settles in seconds
# or is broken. Waiting past that hides the fault instead of reporting it.
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  _waited=0
  while :; do
    GLOO_SOCKET_IFNAME=$(ip -o -4 addr show 2>/dev/null \
        | awk -v ip="$VLLM_HOST_IP" '$4 ~ "^"ip"/" {print $2; exit}')
    [[ -n "$GLOO_SOCKET_IFNAME" ]] && break
    if (( _waited >= ${ROCE_SETTLE_S:-60} )); then
      echo "FATAL: no interface holds VLLM_HOST_IP=$VLLM_HOST_IP after ${_waited}s" >&2
      ip -br addr show >&2; exit 1
    fi
    sleep 5; _waited=$(( _waited + 5 ))
  done
fi
export GLOO_SOCKET_IFNAME

# --- fabric ports: which RoCE devices carry the cluster, at which GID -------
# FABRIC_SUBNETS names one address prefix per cabled port. A second entry puts
# NCCL on both PCIe roots of the ConnectX-7, which ib_write_bw measured at 196
# Gb/s against 112 for one root alone. It is opt-in because the second root's
# registrations are GPU-resident and land AFTER vLLM profiles, so they eat the
# headroom a long prefill needs rather than the KV cache: allocations sat at
# 111.41 GiB against the 104.6 GiB GPU_MEM_UTIL=0.86 budgets. At TP=2 that
# bought nothing -- the per-token allreduce is ~720 KB, a fraction of a
# millisecond against a 45 ms token -- so weigh it only at TP>2.
#
# Keyed on the subnet rather than on VLLM_HOST_IP because the two are no longer
# the same address. mentat identifies a node by its LAN address and the agent
# and daemon must agree on one string, so VLLM_HOST_IP is the LAN one, and no
# RoCE GID will ever match it.
FABRIC_SUBNETS="${FABRIC_SUBNETS:-$CLUSTER_SUBNET}"

# Echoes "<rdma-device> <gid-index>" for the port holding an address in $1.
# Returns 1 when this node has not cabled that port, or its GID has yet to
# appear.
fabric_port() {
  local prefix="$1" found addr ifname dev="" hex i t g d n
  found=$(ip -o -4 addr show 2>/dev/null \
      | awk -v p="$prefix" '$4 ~ "^"p {split($4,a,"/"); print $2, a[1]; exit}')
  [[ -n "$found" ]] || return 1
  ifname="${found%% *}"; addr="${found##* }"
  for d in /sys/class/infiniband/*; do
    for n in "$d"/ports/1/gid_attrs/ndevs/*; do
      [[ -f "$n" ]] || continue
      [[ "$(cat "$n" 2>/dev/null)" == "$ifname" ]] || continue
      dev="$(basename "$d")"; break 2
    done
  done
  [[ -n "$dev" ]] || return 1
  # Slots 0/1 hold the driver's MAC-derived GIDs whatever the IP configuration,
  # so match this port's own static address in its IPv4-mapped form, v2 only.
  hex=$(printf '%s' "$addr" | awk -F. '{printf "%02x%02x:%02x%02x", $1,$2,$3,$4}')
  for i in $(seq 0 15); do
    t=$(cat "/sys/class/infiniband/$dev/ports/1/gid_attrs/types/$i" 2>/dev/null) || continue
    g=$(cat "/sys/class/infiniband/$dev/ports/1/gids/$i" 2>/dev/null) || continue
    [[ "$t" == "RoCE v2" && "$g" == *"ffff:$hex" ]] || continue
    printf '%s %s\n' "$dev" "$i"; return 0
  done
  return 1
}

# There is no correct constant for the GID index. The table is keyed by
# (address, RoCE version), slots are allocated first-free and freed in place, so
# the index for one address differs per node AND per boot. Observed, not
# theorised: on 2026-08-26 one node held the right entry at 6 and its peer at
# 5, because a `nmcli con delete` / `add` had left the first a hole at slot 3. Reboot
# it with the static profile already in place and the table comes up dense,
# moving that 6 to 5, at which point a pinned 6 names an empty slot and NCCL
# fails every TP init with "unhandled system error".
if [[ -z "${NCCL_IB_HCA:-}" || -z "${NCCL_IB_GID_INDEX:-}" ]]; then
  _waited=0
  while :; do
    _hcas=""; _gid=""; _mismatch=""
    for _p in $FABRIC_SUBNETS; do
      _r=$(fabric_port "$_p") || continue
      _d="${_r%% *}"; _i="${_r##* }"
      [[ -n "$_gid" && "$_i" != "$_gid" ]] && _mismatch="$_d at $_i, expected $_gid"
      _gid="${_gid:-$_i}"
      _hcas="${_hcas:+$_hcas,}$_d"
    done
    [[ -n "$_hcas" ]] && break
    if (( _waited >= ${ROCE_SETTLE_S:-60} )); then
      echo "FATAL: no RoCE v2 GID for any of: $FABRIC_SUBNETS" >&2
      ip -br addr show >&2
      ls /sys/class/infiniband/ >&2 || echo "(no /sys/class/infiniband at all)" >&2
      exit 1
    fi
    sleep 5; _waited=$(( _waited + 5 ))
  done
  # NCCL_IB_GID_INDEX applies to every device in NCCL_IB_HCA, so a device whose
  # index differs gets asked for a GID it does not have and QP setup dies with
  # "local GID ::". One IPv4 per fabric interface keeps them aligned; refuse the
  # list rather than hand NCCL one that cannot work.
  # A ring gives each cable its own subnet, and fabric_ring.py picks each
  # device's GID by address instead.
  if [[ -n "$_mismatch" && "$FABRIC_LAYOUT" == mesh ]]; then
    echo "FATAL: fabric ports disagree on GID index ($_mismatch)." >&2
    echo "Each fabric interface must carry exactly one IPv4 address." >&2
    exit 1
  fi
  NCCL_IB_HCA="${NCCL_IB_HCA:-$_hcas}"
  NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-$_gid}"
  echo "fabric: NCCL_IB_HCA=$NCCL_IB_HCA gid=$NCCL_IB_GID_INDEX (subnets: $FABRIC_SUBNETS)"
else
  echo "fabric: NCCL_IB_HCA=$NCCL_IB_HCA gid=$NCCL_IB_GID_INDEX (both pinned; derivation skipped)"
fi
export NCCL_IB_HCA NCCL_IB_GID_INDEX
# The EXACT interface holding VLLM_HOST_IP, not a prefix. This is only the
# out-of-band bootstrap path -- the IB devices above carry the data -- and a
# prefix matches every interface that happens to share it. Once the fabric
# grew a second cable, "enp1s0f" matched both the head link and an unrelated
# one, and NCCL bootstrapped toward the wrong wire: "Connection closed by
# remote peer spark-head".
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}"

# Channel count. NCCL picks 64 here, while every published GB10 recipe pins
# 4-12. 8 was chosen on 2026-09-06 while the fabric was stuck at 12 Gb/s, before
# a power drain fixed it; NCCL's own choice has not been re-tested since.
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
# INFO prints the devices NCCL actually selected, which is the only way to
# confirm both roots are in use.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# --- preflight: things that make the stack slow or fragile without failing --
# One table, WARN rows first to read, never fatal. PREFLIGHT=0 skips it.
preflight() {
  local rows=() warns=0 dev nd mtu rate state phys cur max tp kv need avail swap fs free apps
  # Fields are split on the unit separator: fix text can hold "|" (echo 3 | sudo tee).
  local sep=$'\x1f'
  row() { rows+=("$1$sep$2$sep$3"); [[ "$3" == WARN ]] && warns=$(( warns + 1 )); return 0; }
  local -a devs; IFS=, read -r -a devs <<< "$(sed -E 's/^[=^]+//; s/:[0-9]+//g' <<< "$NCCL_IB_HCA")"
  (( ${#devs[@]} >= 2 )) && row "fabric devices" "${NCCL_IB_HCA}" ok \
    || row "fabric devices" "${NCCL_IB_HCA} (one PCIe root tops out near 110 Gb/s; set FABRIC_SUBNETS to both)" WARN
  local roots; roots=$(for dev in "${devs[@]}"; do basename "$(readlink -f "/sys/class/infiniband/$dev/device")" | cut -d: -f1-2; done | sort -u | wc -l)
  (( ${#devs[@]} < 2 || roots >= 2 )) || row "PCIe roots" "all fabric devices share one root" WARN
  for dev in "${devs[@]}"; do
    local p=/sys/class/infiniband/$dev
    [[ -d $p ]] || { row "$dev" "no such device" WARN; continue; }
    nd=$(ls "$p/device/net" 2>/dev/null | head -1)
    state=$(cut -d' ' -f2 "$p/ports/1/state"); phys=$(cut -d' ' -f2- "$p/ports/1/phys_state")
    [[ $state == ACTIVE ]] && row "$dev link" "$state, $phys" ok || row "$dev link" "$state, $phys" WARN
    rate=$(cut -d' ' -f1 "$p/ports/1/rate")
    (( rate >= 200 )) && row "$dev rate" "$(cat "$p/ports/1/rate")" ok || row "$dev rate" "$(cat "$p/ports/1/rate") (expected 200 Gb/sec)" WARN
    # RoCE's path MTU is the largest IB MTU that fits the netdev's: 9000 gives
    # 4096, 1500 gives 1024. arx and arxbig follow it, and smaller packets cost
    # bandwidth everywhere.
    mtu=$(cat "/sys/class/net/$nd/mtu" 2>/dev/null || echo 0)
    (( mtu >= 4200 )) && row "$nd MTU" "$mtu" ok || row "$nd MTU" "$mtu (RoCE path MTU below 4096; set 9000 end to end)" WARN
    cur="$(cut -d' ' -f1-2 "$p/device/current_link_speed") x$(cat "$p/device/current_link_width")"
    max="$(cut -d' ' -f1-2 "$p/device/max_link_speed") x$(cat "$p/device/max_link_width")"
    [[ $cur == "$max" ]] && row "$dev PCIe" "$cur" ok \
      || row "$dev PCIe" "$cur of $max (degraded link; a full power drain has fixed this before)" WARN
    # Each up/down is two changes; more than a few since boot means a flapping link.
    local flaps; flaps=$(cat "/sys/class/net/$nd/carrier_changes" 2>/dev/null || echo 0)
    (( flaps <= 4 )) || row "$nd flaps" "$flaps carrier changes since boot (check the cable or transceiver)" WARN
    local c v bad=""
    for c in local_ack_timeout_err packet_seq_err out_of_sequence; do
      v=$(cat "$p/ports/1/hw_counters/$c" 2>/dev/null || echo 0)
      (( v > 0 )) && bad+="$c=$v "
    done
    [[ -z $bad ]] || row "$dev retransmits" "${bad% } (since driver load; drops, likely no PFC)" info
  done
  # RDMA pins its buffers; a memlock limit fails registration or NCCL's setup.
  [[ "$(ulimit -l)" == unlimited ]] && row "memlock limit" "unlimited" ok \
    || row "memlock limit" "$(ulimit -l) KiB (RDMA registration needs unlimited: ulimits memlock -1)" WARN
  # GPU: a power-delivery or thermal event caps clocks for the whole boot.
  local ev; ev=$(nvidia-smi --query-gpu=clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,temperature.gpu --format=csv,noheader 2>/dev/null)
  if [[ -z $ev ]]; then row "GPU" "nvidia-smi unavailable" WARN
  elif awk -F', ' '{for (i = 1; i < NF; i++) if ($i == "Active") f = 1} END {exit !f}' <<< "$ev"; then
    row "GPU slowdown" "$ev (hw, power brake, hw thermal, sw thermal, C)" WARN
  else row "GPU slowdown" "none, ${ev##*, } C" ok; fi
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')
  [[ -z $apps ]] && row "other GPU processes" "none" ok || row "other GPU processes" "${apps%;}" WARN
  # Host memory: ~182 GiB of weights split TP ways, the pinned KV, and ~30 GiB
  # more (CUDA graphs, the activation reserve, NCCL and RDMA buffers, the
  # processes themselves). TP=2 with 8 GiB of KV came to ~123 GiB and the OOM
  # killer took a worker mid-prefill.
  tp=${TP:-4}; kv=$(( ${KV_CACHE_MEMORY:-27917287424} >> 30 ))
  need=$(( 182 / tp + kv + 30 ))
  avail=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) >> 20 ))
  (( avail >= need )) && row "host memory" "${avail} GiB free, ~${need} GiB needed" ok \
    || row "host memory" "${avail} GiB free, ~${need} GiB needed (stop other models; lower KV_CACHE_MEMORY)" WARN
  # MemAvailable counts the page cache as free, but on GB10 CUDA allocations
  # can fail before the kernel reclaims it: NCCL's init died "out of memory"
  # with ~113 GB cached after a weight copy. Drop the model files' own cached
  # pages (no privilege needed), then say how to drop the rest.
  unfree() { echo $(( $(awk '/MemFree/ {print $2}' /proc/meminfo) >> 20 )); }
  if (( $(unfree) < need )); then
    python3 - "$MODEL" "${DFLASH_MODEL:-/models/glm-5.3-flash-dflash2}" "${VLLM_WEIGHT_SNAPSHOT_DIR:-}" <<'PY' 2>/dev/null || true
import os, sys
for root in filter(None, sys.argv[1:]):
    for dirpath, _, files in os.walk(root):
        for name in files:
            try:
                fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass
PY
  fi
  local memfree; memfree=$(unfree)
  local cached; cached=$(( $(awk '/^Cached:/ {print $2}' /proc/meminfo) >> 20 ))
  (( memfree >= need )) && row "page cache" "${cached} GiB cached, ${memfree} GiB free" ok \
    || row "page cache" "${cached} GiB cached, ${memfree} GiB free, ~${need} GiB needed (GPU allocations may fail: on the host, sync; echo 3 | sudo tee /proc/sys/vm/drop_caches)" WARN
  swap=$(( ( $(awk '/SwapTotal/ {print $2}' /proc/meminfo) - $(awk '/SwapFree/ {print $2}' /proc/meminfo) ) >> 20 ))
  (( swap < 1 )) || row "swap in use" "${swap} GiB (the GPU shares this memory; paging stalls it)" WARN
  fs=$(findmnt -n -o FSTYPE -T "${MODEL_DIR:-/models/glm-5.3-flash-nvfp4}" 2>/dev/null)
  [[ $fs == nfs* || $fs == cifs || $fs == fuse* ]] && row "model filesystem" "$fs (every rank reads it all; use local disk)" WARN \
    || row "model filesystem" "${fs:-unknown}" ok
  # The first boot at a TP size writes a weight snapshot (snapshot.yaml).
  if [[ -n "${VLLM_WEIGHT_SNAPSHOT_DIR:-}" ]]; then
    free=$(df -BG --output=avail "${CACHE_ROOT:-/root/.cache}" 2>/dev/null | tail -1 | tr -dc 0-9)
    ver=$(sed -n 's/^SNAPSHOT_VERSION = \([0-9]*\).*/\1/p' \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_snapshot.py 2>/dev/null)
    if compgen -G "$VLLM_WEIGHT_SNAPSHOT_DIR/v${ver}-target-tp*of${tp}-*" >/dev/null; then row "weight snapshot" "present for TP=$tp" ok
    elif (( ${free:-0} >= 182 / tp + 5 )); then row "weight snapshot" "none yet for TP=$tp, ${free} GiB free" ok
    else row "weight snapshot" "none yet for TP=$tp, ${free:-?} GiB free, needs ~$(( 182 / tp )) GiB" WARN; fi
  fi
  # A desktop session holds GPU memory and CPU on the shared memory, and these
  # boxes ship with GDM enabled. systemd lists running units in /run/systemd/units.
  if [[ -d /host/systemd-units ]]; then
    local dm; dm=$(ls /host/systemd-units 2>/dev/null \
      | sed -n 's/^invocation:\(gdm3\?\|lightdm\|sddm\|display-manager\)\.service$/\1/p' | head -1)
    [[ -z $dm ]] && row "desktop session" "none" ok \
      || row "desktop session" "$dm is running (sudo systemctl set-default multi-user.target && sudo systemctl isolate multi-user.target)" WARN
  fi
  grep -q '^search \.$' /etc/resolv.conf 2>/dev/null && row "resolv.conf" "'search .' frozen in; bare hostnames fail (restart the container)" WARN
  local on; on=$(overlays | tr '\n' ' ')
  [[ -n $on ]] && row "overlays" "$on" ok || row "overlays" "none: stock kernels" WARN
  echo "preflight ($warns warning$([[ $warns == 1 ]] || echo s)):"
  printf '%s\n' "${rows[@]}" | awk -F"$sep" '$3 == "WARN" {printf "  %-5s %-26s %s\n", $3, $1, $2}'
  printf '%s\n' "${rows[@]}" | awk -F"$sep" '$3 != "WARN" {printf "  %-5s %-26s %s\n", $3, $1, $2}'
}

# The experimental/ overlays this container has mounted, one name per line.
# Each one mounts a file the image does not ship, except sp, which only sets
# its environment.
overlays() {
  local v=/usr/local/lib/python3.12/dist-packages/vllm
  [[ -e $v/distributed/device_communicators/arx.py ]] && echo arx
  [[ -e $v/model_executor/model_loader/weight_snapshot.py ]] && echo snapshot
  [[ -e $v/v1/core/sched/adaptive_k.py ]] && echo adaptive-k
  [[ -e $v/model_executor/model_loader/dense_fp8.py ]] && echo fp8
  [[ -e $v/model_executor/layers/fused_moe/megamoe_vllm.py ]] && echo megamoe
  [[ -e $v/v1/attention/backends/mla/gb10_sparse_mla.py ]] && echo fixes
  [[ -n "${VLLM_GLM_SP_TP:-}" ]] && echo sp
  [[ -e $v/models/glm5next/common/recoverssm.py ]] && echo recoverssm
  return 0
}
[[ "$NATIVE_DRY_RUN" != 1 && "${PREFLIGHT:-1}" == 1 ]] && preflight || true

# Without any overlay the stack runs vLLM's stock kernels at about half the
# README's decode speed, and nothing else says so. Refuse to start instead.
if [[ -z "$(overlays)" && "${ALLOW_STOCK:-0}" != 1 ]]; then
  echo "FATAL: no experimental overlays are mounted, so this would run stock kernels at about half the" \
       "README's speed. Start the stack with ./glm53 up -d (README step 5), or set ALLOW_STOCK=1" \
       "in compose/.env to run stock on purpose." >&2
  exit 1
fi

# --- worker memory ---------------------------------------------------------
# Torch's caching allocator never hands a freed block back to the OS, and on
# unified memory that block is host memory. The sparse indexer scores
# chunk x context/kpool, so a session whose context keeps growing asks for a
# slightly bigger block each step and strands the last one.
# expandable_segments:True grows one segment instead of laddering.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Hard ceiling on one worker's device memory, as a fraction of the 121.6 GiB,
# read by spark_mem_trace.py. Keep it above GPU_MEM_UTIL by enough for a full
# prefill's indexer scratch, and below 1.0 by whatever the host needs to keep
# forking: past this a request raises OutOfMemoryError instead of the node
# wedging. 0.92 is ~111.9 GiB against GPU_MEM_UTIL's ~107.
export TORCH_MEM_FRACTION="${TORCH_MEM_FRACTION:-0.92}"

# The pinned vLLM MultiprocExecutor coordinates both fixed ranks directly.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset RAY_ADDRESS VLLM_USE_RAY_V2_EXECUTOR_BACKEND

# --- scheduler: per-step token budget, and one prefill's share of it --------
# One setting, not two. The scheduler gives a long prefill
# min(threshold, remaining budget), so a threshold at or above the budget lets
# one big prompt take the entire step -- no decodes of running requests, no
# admissions from the waiting queue. Measured on DS4 2026-08-25: a 90-token
# "ping" sent 20s into a 198K-token prefill took 125.5s to answer.
#
# 16384 with each prefill capped at 2304, so one step carries several prefill
# chunks and every running decode. Measured on TP=4 (2026-09-06): 8192 and
# 16384 prefill a 200k prompt in the same time (234.1 s against 237.7 s); the
# old 9.5x win for 8192 was a PP=2 pipeline bubble, gone at PP=1. Measure this
# at 200k, never at 8k: an 8k prompt fits inside both budgets.
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

# --- max_num_seqs is capped by the KDA state, not by throughput -------------
# 34 of this model's 45 layers are KDA linear attention, and a linear-attention
# layer keeps a fixed-size RECURRENT STATE per sequence rather than a per-token
# KV entry. vLLM accounts for those as "Mamba cache blocks", one per decode
# sequence, and they are allocated out of whatever is left after weights.
#
# Measured here 2026-08-26: exactly 32 blocks fit at GPU_MEM_UTIL=0.86 with the
# 181 GiB NVFP4 checkpoint. vLLM's default max_num_seqs of 256 therefore aborts
# at CUDA graph capture, AFTER a full 10-minute weight load:
#
#   ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (32).
#   Each decode sequence requires one Mamba cache block, so CUDA graph capture
#   cannot proceed.
#
# This is a hard structural cap, not a tuning preference -- raising
# GPU_MEM_UTIL is the only way to buy more blocks. The default takes all 32.
#
# Keep CUDAGRAPH_CAPTURE_SIZES's largest entry >= this value: with MTP off a
# decode step is one token per sequence, so a full batch is exactly this many
# tokens, and a step larger than the biggest captured size runs uncaptured.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

# One prefill's share of the step: 2304, one KDA block. A multiple of the block
# matters -- 2048 produced alternating 2048/256 chunks. With it (2026-09-06),
# prefill ran 2,004 tok/s at 200k and a 12-token ping sent under a 120k prefill
# answered in 4.83 s. DECODE_RESERVE_TOKENS, when set, replaces it with
# budget minus reserve, which is how DS4 expresses the same split.
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-2304}"
if [[ -n "${DECODE_RESERVE_TOKENS:-}" ]]; then
  LONG_PREFILL_TOKEN_THRESHOLD=$(( (MAX_NUM_BATCHED_TOKENS - DECODE_RESERVE_TOKENS) / 4 * 4 ))
  if (( LONG_PREFILL_TOKEN_THRESHOLD < 4 )); then
    echo "FATAL: DECODE_RESERVE_TOKENS=${DECODE_RESERVE_TOKENS} leaves only" >&2
    echo "${LONG_PREFILL_TOKEN_THRESHOLD} tokens for prefill out of a ${MAX_NUM_BATCHED_TOKENS} budget." >&2
    exit 1
  fi
fi
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-8 16 32 64 96 128 192 256}"
echo "scheduler: budget=${MAX_NUM_BATCHED_TOKENS} prefill<=${LONG_PREFILL_TOKEN_THRESHOLD}" \
     "reserve=$(( MAX_NUM_BATCHED_TOKENS - LONG_PREFILL_TOKEN_THRESHOLD ))/step"

export MAX_JOBS="${MAX_JOBS:-4}"

# --- JIT/autotune cache tiers ----------------------------------------------
# Two tiers, because the caches have different invalidation domains: the shared
# tier is keyed on version/arch/kv-dtype only and is the expensive one to
# rebuild, so it must stay stable across serving-option experiments; the keyed
# tier holds torch.compile, which vLLM hashes against the whole serving config.
CACHE_ROOT="${CACHE_ROOT:-/root/.cache}"
_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
_arch="${_arch:-unknown}"
# --- KV cache dtype: auto (bf16), NOT fp8 -----------------------------------
# fp8 here selects vLLM's `fp8_ds_mla` packed KV format, which is DeepSeek's and
# hardcodes pe_dim == 64:
#
#   RuntimeError: concat_and_cache_mla, cache_kernels.cu:866,
#   pe_dim must be 64 for fp8_ds_mla
#
# GLM-5.3-Flash is NoPE sparse MLA -- `qk_rope_head_dim: 0`, no rotary on the
# sparse path at all -- so pe_dim is 0 and that kernel refuses. This is what the
# vendor recipe means by "FlashInfer 0.6.17+ is required for NoPE sparse MLA":
# the NoPE variant is a different code path, not just a newer one.
#
# Copying ds4-flash's `--kv-cache-dtype fp8` across is therefore wrong: DS4 is
# DeepSeek geometry (pe_dim 64) and this model is not. It costs a full weight
# load (~10 min) to find out, because the failure is at KV cache init.
#
# The default below is fp8_e4m3, and it is what every measurement on this model
# was taken with: the Korean corruption probe comes back clean and 200k recall
# passes on it. Only 11 of 45 layers carry a KV cache and they use
# kv_lora_rank 512, so bf16 would also fit -- but nothing here has been measured
# on bf16, so do not switch on the assumption that it is the safer default.
_kv="${KV_CACHE_DTYPE:-fp8_e4m3}"
export SHARED_TAG="${SHARED_TAG:-glm53-${_arch}-${_kv}}"
# Resolve ONCE, above every use. Two MAX_MODEL_LEN defaults in one
# file is the "two places to change and one silently winning" trap the DS4
# compose file warns about -- and it was live here: the cache tag said 131072
# while the server was told 262144.
#
# 524288 of the model's 1M. KV is cheap here -- only 11 of 45 layers carry one,
# at kv_lora_rank 512 -- so the 26 GiB pin below holds 2,632,595 tokens with
# DFlash2, 5.02 requests at full length (head boot log, 2026-09-23).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
_optstr="tp=${TP} mtp=${MTP} spec=${SPEC_METHOD} len=${MAX_MODEL_LEN} cg=${CUDAGRAPH_CAPTURE_SIZES}"
_opthash=$(printf '%s' "$_optstr" | sha256sum | cut -c1-8)
export CACHE_TAG="${CACHE_TAG:-${SHARED_TAG}-${_opthash}}"

# FlashInfer autotune: EPHEMERAL on TP>1, and wiped on every start. A persisted
# cache deadlocks the NEXT boot, silently and forever. On the nightly, rank 0
# reads its cache and broadcasts it, but only rank 0 ever saves, and the fused-
# MoE entries it saves key per rank: ranks 1-3 miss what rank 0 hits, go off to
# profile (a CPU-group all_reduce) while rank 0 has moved on to the model's NCCL
# all_reduce, and each waits for the other. Measured 2026-09-23 with py-spy on
# spark-glm53:v3. The first boot always works (nobody has a cache); every boot
# after a good one hangs with the head at 96% GPU and the workers at 0%.
# Wiping on start, not just keeping it out of the bind mount, matters: a head
# engine restart keeps the container's /tmp. Costs ~2 min of autotune a boot.
# AUTOTUNE_CACHE=persist brings the old behaviour back (TP=1 is safe with it).
if [[ "${AUTOTUNE_CACHE:-}" == "persist" || "${TP:-1}" -le 1 ]]; then
  export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${CACHE_ROOT}/${SHARED_TAG}/flashinfer_autotune"
else
  export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/flashinfer_autotune
  [[ "$NATIVE_DRY_RUN" == 1 ]] || rm -rf "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"
fi
echo "autotune cache: $VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/${CACHE_TAG}/vllm"
[[ "$NATIVE_DRY_RUN" == 1 ]] || mkdir -p "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" "$VLLM_CACHE_ROOT"

# FlashInfer's own JIT cache is deliberately NOT redirected. FLASHINFER_CACHE_DIR
# looks like the knob but is a module-level constant computed from
# FLASHINFER_WORKSPACE_BASE (default $HOME), so setting it does nothing at all.
# $HOME/.cache is the bind mount, so the default already persists.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/${SHARED_TAG}/triton}"
[[ "$NATIVE_DRY_RUN" == 1 ]] || mkdir -p "$TRITON_CACHE_DIR"

# TileLang JIT builds the mHC (Manifold-Constrained Hyper-Connection) kernels
# this architecture adds, and the DSA/MTP kernels. Its default is
# ~/.tilelang/cache -- a SIBLING of the bind-mounted ~/.cache, so left alone it
# lands inside the container and is destroyed on every `compose down`, and every
# boot recompiles. Same invalidation domain as FlashInfer, hence the shared tier.
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${CACHE_ROOT}/${SHARED_TAG}/tilelang}"
[[ "$NATIVE_DRY_RUN" == 1 ]] || mkdir -p "$TILELANG_CACHE_DIR"
echo "cache shared: ${CACHE_ROOT}/${SHARED_TAG} (arch=${_arch} kv=${_kv})"
echo "cache keyed : ${CACHE_ROOT}/${CACHE_TAG} ($_optstr)"
echo "cache tilelang: $TILELANG_CACHE_DIR"

# --- status page ------------------------------------------------------------
# Own port, started immediately. Loading 181 GiB of weights takes long enough
# that "connection refused" is a poor answer to "what is it doing?", and this
# model's metrics read zero throughout a long prefill anyway.
export STAGE_FILE="${STAGE_FILE:-/tmp/glm53-stage}"
stage() {
  [[ "$NATIVE_DRY_RUN" == 1 ]] || echo "$1" >> "$STAGE_FILE"
  echo "== stage: $1 =="
}
if [[ "$NATIVE_DRY_RUN" != 1 ]]; then
  : > "$STAGE_FILE"
  STATUS_PORT="${STATUS_PORT:-8082}" PORT="${API_PORT:-8000}" \
    python3 /usr/local/bin/status-server.py &
fi
stage starting

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "FATAL: no config.json under MODEL_DIR=$MODEL." >&2
  echo "Both ranks read the full checkpoint, so it must be present on THIS node." >&2
  exit 1
fi

if [[ "$NATIVE_DRY_RUN" != 1 && -d "${MCP_LOG_DIR:-/logs}" && -w "${MCP_LOG_DIR:-/logs}" ]]; then
  # An elected box has no role yet, and its role can change between boots.
  _log="${MCP_LOG_DIR:-/logs}/vllm${ROLE:+-$ROLE}.log"
  [[ -f "$_log" ]] && mv -f "$_log" "${_log%.log}.prev.log" 2>/dev/null || true
  echo "logging to $_log"
  exec > >(tee "$_log") 2>&1
fi

# Reuse the existing fabric diagnostic; it rendezvouses directly via TCP/NCCL.
if [[ "$NATIVE_DRY_RUN" != 1 && "${FABRIC_CHECK:-1}" == 1 ]]; then
  stage fabric-check
  FABRIC_LAYOUT=mesh timeout "$(( 3 * ${FABRIC_CHECK_TIMEOUT_S:-120} ))" \
    python3 /usr/local/bin/fabric-check.py || echo 'fabric check: failed or timed out; continuing'
fi

# The drafter was chosen at the top (SPEC_METHOD). A missing drafter would
# otherwise fail only after the ~10 minute target load, so check it first.
SPEC=()
case "$SPEC_METHOD" in
  dflash)
    SPEC_TOKENS="${SPEC_TOKENS:-7}"
    DFLASH_MODEL="${DFLASH_MODEL:-/models/glm-5.3-flash-dflash2}"
    if [[ ! -f "$DFLASH_MODEL/config.json" ]]; then
      echo "FATAL: SPEC_METHOD=dflash but no config.json under DFLASH_MODEL=$DFLASH_MODEL." >&2
      echo "Mount the DFlash2 drafter there on every node, or set SPEC_METHOD=mtp." >&2
      exit 1
    fi
    SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DFLASH_MODEL}\",\"num_speculative_tokens\":${SPEC_TOKENS}}") ;;
  mtp)
    SPEC_TOKENS="${SPEC_TOKENS:-4}"
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}") ;;
  none) ;;
  *) echo "FATAL: SPEC_METHOD=$SPEC_METHOD; want mtp, dflash or none" >&2; exit 1 ;;
esac
# SPEC_EXTRA: more keys for the speculative config above, as one JSON object.
# adaptive-k.yaml puts its per-batch-size draft widths, the drafter's attention
# backend and disable_eagle_block_drop here. They used to arrive as a second
# --speculative-config in EXTRA_ARGS, and vLLM keeps the last one, so whenever
# that overlay was loaded SPEC_METHOD, SPEC_TOKENS and DFLASH_MODEL were
# silently ignored (SPEC_METHOD=mtp still served DFlash2). Draft widths above
# SPEC_TOKENS are capped to it.
if [[ ${#SPEC[@]} -gt 0 && "${EXTRA_ARGS:-}" == *--speculative-config* ]]; then
  echo "FATAL: EXTRA_ARGS sets --speculative-config too. vLLM keeps the last one, so SPEC_METHOD," \
       "SPEC_TOKENS and DFLASH_MODEL would be ignored. Put extra keys in SPEC_EXTRA (a JSON object) instead." >&2
  exit 1
fi
if [[ -n "${SPEC_EXTRA:-}" ]]; then
  if [[ "$SPEC_METHOD" != dflash ]]; then
    echo "FATAL: SPEC_EXTRA is set (adaptive-k.yaml sets it, tuned for DFlash2) but SPEC_METHOD=$SPEC_METHOD." \
         "Start without experimental/compose/adaptive-k.yaml to run $SPEC_METHOD." >&2
    exit 1
  fi
  SPEC[1]=$(python3 -c '
import json, sys
spec, extra = json.loads(sys.argv[1]), json.loads(sys.argv[2])
if not isinstance(extra, dict): sys.exit("SPEC_EXTRA must be a JSON object")
fixed = sorted(set(extra) & {"method", "model", "num_speculative_tokens"})
if fixed: sys.exit("SPEC_EXTRA must not set " + ", ".join(fixed) + ": use SPEC_METHOD, DFLASH_MODEL and SPEC_TOKENS")
n = spec["num_speculative_tokens"]
for row in extra.get("num_speculative_tokens_per_batch_size", []): row[2] = min(row[2], n)
spec.update(extra)
print(json.dumps(spec, separators=(",", ":")))' "${SPEC[1]}" "$SPEC_EXTRA") || { echo "FATAL: SPEC_EXTRA: see above ($SPEC_EXTRA)" >&2; exit 1; }
fi
[[ ${#SPEC[@]} -gt 0 ]] && echo "speculative: ${SPEC[*]}"

stage loading

if [[ "$ROLE" == head && "$NATIVE_DRY_RUN" != 1 ]]; then
if [[ "${SELF_TEST:-1}" == "1" ]]; then
  (
    if python3 /usr/local/bin/self-test.py \
         --base "http://127.0.0.1:${API_PORT:-8002}" --model "$SERVED"; then
      stage serving
    else
      stage self-test-failed
      echo "!! SELF-TEST FAILED. Serving anyway so the model can be probed;" >&2
      echo "!! the status page reports unhealthy. A wrong-but-healthy model is" >&2
      echo "!! the specific failure this gate exists to catch." >&2
    fi
  ) &
else
  ( for _ in $(seq 1 480); do
      curl -sf -o /dev/null --max-time 4 "http://127.0.0.1:${API_PORT:-8002}/v1/models" \
        && { stage serving; break; }
      sleep 5
    done ) &
fi

fi

# --- tool parser --------------------------------------------------------------
# The stock glm47 by default. TOOL_PARSER=glm47_failclosed selects the
# fail-closed plugin baked at /usr/local/share (loaded with
# --tool-parser-plugin), a stopgap for tool calls corrupted upstream. It
# refuses any call to a tool the request did not list, which stalls clients
# that reach deferred tools through a tool-search step (Qwen Code's MCP
# tools). The stock parser drops such a call silently: this vLLM fixes
# validate_tool_names=True in glm47_moe_config, with no flag to turn it off.
# Selected here, not through EXTRA_ARGS: an override restating EXTRA_ARGS once
# dropped the plugin from production.
TOOL_PARSER="${TOOL_PARSER:-glm47}"
TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
if [[ "$TOOL_PARSER" == "glm47_failclosed" ]]; then
  TOOL_ARGS=(--tool-parser-plugin /usr/local/share/glm47_failclosed.py "${TOOL_ARGS[@]}")
fi
echo "tool parser: $TOOL_PARSER"

# --- sparse indexer top-k -------------------------------------------------------
# GB10 cannot run persistent_topk once the KV pool passes ~3.4M tokens (it would
# oversubscribe 48 CTAs), and the FilteredTopK fallback wants 128 KB of shared
# memory per block against the 101,376 B an SM has. per_row is the same
# computation without the persistent-CTA scheme. This used to be a patch
# (gb10_topk_fallback.py); on main it is a flag. "auto" tries cooperative,
# then persistent, then per_row.
TOPK_BACKEND="${TOPK_BACKEND:-per_row}"
TOPK_ARGS=(--sparse-indexer-topk-backend "$TOPK_BACKEND")

# --load-format auto, NOT sharded_state -- see the header.
#
# Parser names do not match the model version, which is normal here: the vendor
# recipe for 5.3-Flash specifies glm47 and glm45.
# --- MoE backend and CUDA graphs -------------------------------------------
# flashinfer_cutlass, the native NVFP4 kernel, which runs W4A4 off the
# checkpoint's per-projection input scales. Production has served on it with
# the nvidia checkpoint since 2026-09-21; why it replaced marlin then is not
# recorded. It decodes 109.8 / 88.8 / 52.6 tok/s (structured / code / prose,
# v6 + DFlash2, 2026-09-23) and passes the thinking-on count-to-200 probe 8/8.
# It takes vLLM's own CUDA graph sizes; CUDAGRAPH_CAPTURE_SIZES applies to
# marlin only.
#
# marlin was the default before that. It is weight-only: it dequantises to FP16
# and runs bf16 activations, so a checkpoint's input scales are never read. It
# was adopted when the native kernels died on sm_121 with
# cudaErrorNoKernelImageForDevice, a failure both MiaAI-Lab and LibertAI later
# put down to one checkpoint's uninitialised input_scale. It is also not
# bit-deterministic at M >> 1 (dev/TOPK-CORRUPTION.md).
#
# Pairing marlin with --enforce-eager is recipe convention, not a correctness
# constraint. Marlin's workspace helper reuses storage precisely so a captured
# graph's addresses stay valid, and the only eager gate in this vLLM is for
# DeepseekV4. Capture measures 6 s here. CUDA_GRAPHS=0 restores eager.
#
# Capture sizes are in TOKENS and vLLM rounds them to multiples of (1 + draft
# tokens). Above max_num_seqs * (1 + k) the dispatcher returns NONE and decode
# falls back to eager with nothing in the log to say so, which is why the
# ceiling is computed and checked rather than left to whoever edits the list.
MOE=()
MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}"
if [[ "$MOE_BACKEND" == "marlin" ]]; then
  if [[ "${CUDA_GRAPHS:-1}" == "1" ]]; then
    _k=1
    [[ -n "$SPEC_METHOD" && "$SPEC_METHOD" != none ]] && _k=$(( SPEC_TOKENS + 1 ))
    _ceil=$(( MAX_NUM_SEQS * _k ))
    _largest=$(tr ' ' '\n' <<<"$CUDAGRAPH_CAPTURE_SIZES" | sort -n | tail -1)
    if (( _largest > _ceil )); then
      echo "WARNING: largest capture size $_largest exceeds max_num_seqs*(1+k)=$_ceil;" >&2
      echo "decode above $_ceil tokens will run eager and log nothing." >&2
    fi
    MOE=(--moe-backend marlin
         --cudagraph-capture-sizes ${CUDAGRAPH_CAPTURE_SIZES}
         --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}\"}")
    echo "MoE backend: marlin, cudagraphs ${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}" \
         "(sizes: $CUDAGRAPH_CAPTURE_SIZES, ceiling ${_ceil})"
  else
    MOE=(--moe-backend marlin --enforce-eager)
    echo "MoE backend: marlin (enforce-eager, CUDA_GRAPHS=0)"
  fi
else
  MOE=(--moe-backend "${MOE_BACKEND}")
  # vLLM captures graphs up to max_num_seqs x (1 + drafts) x 2, capped at
  # 512 tokens. The top half only serves steps that mix decodes with a short
  # new prompt, so stop at the largest decode step: 32 tokens at TP=2 instead
  # of 64, for example. Those mixed steps run without a graph.
  # A draft width that varies with batch size (adaptive-k.yaml's
  # num_speculative_tokens_per_batch_size, through SPEC_EXTRA) decodes 1 + k tokens per request
  # for each k in the schedule: 3, 5, 6, 10, 12, 15. vLLM adds those sizes to
  # its capture list only when it picks the ceiling itself, so an explicit cap
  # leaves them to run without a full graph. Leave the ceiling to vLLM there,
  # unless CUDAGRAPH_MAX sets one.
  if [[ -z "${CUDAGRAPH_MAX:-}" && "${SPEC[*]:-}" == *num_speculative_tokens_per_batch_size* ]]; then
    echo "MoE backend: ${MOE_BACKEND}, vLLM's own CUDA graph sizes (draft width varies with batch size)"
  else
    _q=1; [[ -n "$SPEC_METHOD" && "$SPEC_METHOD" != none ]] && _q=$(( SPEC_TOKENS + 1 ))
    _cgmax=$(( MAX_NUM_SEQS * _q < 512 ? MAX_NUM_SEQS * _q : 512 ))
    CUDAGRAPH_MAX="${CUDAGRAPH_MAX:-$_cgmax}"
    MOE+=(--max-cudagraph-capture-size "$CUDAGRAPH_MAX")
    echo "MoE backend: ${MOE_BACKEND}, CUDA graphs up to ${CUDAGRAPH_MAX} tokens"
  fi
fi

# Multimodal: up to 16 images a prompt, no video. Exceeding a cap is a clean
# 400 with an OpenAI-shaped error body, not an engine fault, but a client that
# treats any 400 as fatal dies on it.
# The closing brace is escaped: bash ends a :- default at the first unescaped
# one, so it would cut the default short and append the tail as literal text.
LIMIT_MM="${LIMIT_MM:-{\"image\":16,\"video\":0\}}"
python3 -c 'import json, sys; json.loads(sys.argv[1])' "$LIMIT_MM" 2>/dev/null || {
  echo "FATAL: LIMIT_MM is not valid JSON: $LIMIT_MM" >&2
  echo "       Write the whole object, closing brace included." >&2
  exit 1; }
MM=(--limit-mm-per-prompt "$LIMIT_MM")
# 0: run the max-size dummy forward at init, so the vision encoder's peak is
# budgeted at startup instead of coming out of the headroom a long prefill
# needs at run time. 1 skips it, which is faster to boot and was the default
# while images were capped at 4.
[[ "${SKIP_MM_PROFILING:-0}" == "1" ]] && MM+=(--skip-mm-profiling)

# The image's template is nvidia's, with one change: thinking off asks for low
# reasoning effort. GLM-5.3-Flash has no non-thinking mode, and the empty
# <think></think> that thinking off otherwise produces makes long output repeat
# and skip (README, Troubleshooting). The checkpoint's own template is ignored,
# so a fresh download cannot bring the problem back. CHAT_TEMPLATE overrides.
TMPL=()
: "${CHAT_TEMPLATE:=/usr/local/share/glm53-chat-template.jinja}"
if [[ -n "$CHAT_TEMPLATE" ]]; then
  TMPL=(--chat-template "$CHAT_TEMPLATE")
  echo "chat template: $CHAT_TEMPLATE"
fi

# A 320B MoE takes far longer to init than the default allows.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

# --- KV cache pin ----------------------------------------------------------
# Left unset, vLLM PROFILES the KV cache and takes everything available -- here
# that was 13.59 GiB, giving 796,361 tokens. FlashInfer's autotune then runs
# during warmup, asks the driver for more, and there is none:
#
#   NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal
#   RayWorkerProc rank=[0] died unexpectedly
#
# On a discrete GPU the profiler's headroom assumptions usually survive this.
# On GB10's unified memory they do not, which is what the upstream recipe means
# by "--kv-cache-memory pin (UMA OOM only)". Pin it below the profiled figure so
# warmup has room.
#
# Measured 2026-08-27: a 10 GiB pin made things WORSE, because skipping the
# profile also discards the gpu_memory_utilization margin autotune relies on.
# That was before TORCH_MEM_FRACTION and the ephemeral autotune cache.
#
# 26 GiB is the pin that serves under DFlash2. At 28 the head sat near 1 GiB
# free and eight concurrent long-context requests had a worker OOM-killed. A
# pin also stops the pool moving by ~0.9 GiB between identical boots with
# page-cache timing at profiling. DFlash2 costs 41% of the pool: 3,437,736
# tokens with speculation off against 2,024,644 with it (pre-nightly image,
# 2026-09-06).
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-27917287424}"
KV_ARGS=(--kv-cache-memory "${KV_CACHE_MEMORY}")
echo "kv-cache-memory pinned to ${KV_CACHE_MEMORY} bytes ($(( KV_CACHE_MEMORY / 1073741824 )) GiB)"

# eager stages each shard through anonymous memory and loads faster -- 511 s
# against 690 s for lazy. Unpinned, its buffers were still resident when vLLM
# profiled for KV and cost 38% of the cache: 810,576 tokens against 1,296,922.
# With the KV pin above the pool is not profiled, and production runs eager.
# Go back to lazy if the pin is ever dropped.
NET=()
if [[ "$ROLE" == head ]]; then
  NET=(--host 0.0.0.0 --port "${API_PORT:-8000}")
else
  NET=(--headless)
fi
CMD=(vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size "$TP" \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$HEAD_HOST" --master-port "${MASTER_PORT:-29553}" \
  --load-format auto \
  --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY:-eager}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.88}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --override-generation-config '{"max_new_tokens":null}' \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
  --block-size "${BLOCK_SIZE:-2304}" \
  "${KV_ARGS[@]}" \
  --trust-remote-code \
  --enable-prefix-caching \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}" \
  "${MOE[@]}" \
  "${MM[@]}" \
  "${TMPL[@]}" \
  ${ITERATION_DETAILS:+--enable-logging-iteration-details} \
  "${TOOL_ARGS[@]}" \
  "${TOPK_ARGS[@]}" \
  --reasoning-parser glm45 \
  "${SPEC[@]}" \
  "${NET[@]}" \
  ${EXTRA_ARGS:-})
if [[ "$NATIVE_DRY_RUN" == 1 ]]; then
  printf '\nNATIVE_COMMAND_JSON='
  printf '%s\0' "${CMD[@]}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.buffer.read().decode().split("\0")[:-1]))'
  exit 0
fi
exec "${CMD[@]}"
