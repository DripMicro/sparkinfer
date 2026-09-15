#!/usr/bin/env bash
# Cron wrapper for the sparkinfer Qwen3.8-27B PR eval bot (the unsloth/Qwen3.8-27B-NVFP4 checkpoint):
#
#   30 * * * * /home/autotiny/Desktop/sparkinfer/eval/run_qwen38_cron.sh >> /tmp/sparkinfer_qwen38_bot.log 2>&1
#
#   Hourly at :30, the slot run_dspark_cron.sh held until 2026-09-15, when the DSpark bot was
#   paused and this bot took its place with concurrent-decode axes added (issue #1080). The Muse
#   Glimmer bot keeps :00. Every bot drives the one pinned GPU, so two must never share a slot.
#
#   A ref is one build plus: the decode@128 / prefill@16k sweep, the concurrent-decode ladder
#   (c1..c32, median of three complete runs per width), the accuracy score dump and the Qwen3.6
#   guard. A round is 1 main baseline + 1 run per pending PR.
#
#   If a round ever DOES overrun the interval (a large PR backlog), the flock below makes the
#   overlapping tick exit 0 after 120s rather than piling up — the effect is a skipped tick, not a
#   queue. Do NOT raise flock's -w to try to queue ticks: that would serialize a backlog of stale
#   rounds against one GPU.
#
# Policy (same as the sibling bots):
#   • Pinned eval box only; never rent / never start from cron when down.
#   • Shares /tmp/sparkinfer_bot.lock with run_dflash_cron.sh (and the AR bot / sparkinfer-web
#     dashboard-sync crons) — CRITICAL: this MUST be the SAME lock file, since every one of these
#     bots drives the ONE pinned GPU on the SAME box and would otherwise race for it if a cron
#     tick overlaps. Wait a bounded amount instead of failing instantly: long enough to outlast a
#     quick sibling-bot tick, short enough to still bail if something is genuinely stuck.
#   • GPU up → full speed eval (prefill@16k, concurrent decode c1..c32, floors) + differential
#     accuracy + Qwen3.6 guard;
#     GPU down → --labels-only (no GPU, no ssh, pure label reconciliation).
#   • Auto-merge follows SPARKINFER_QWEN38_AUTOMERGE, which .env.eval sets to 1 (explicit
#     decision 2026-08-15). This script never forces it.
export HOME="${HOME:-/home/autotiny}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export VAST_NO_AUTO_PROVISION=1

exec 9>/tmp/sparkinfer_bot.lock
flock -w 120 9 || { echo "[$(date -u +%FT%TZ)] lock held 120s+ — previous bot run still active, skipping qwen38 tick"; exit 0; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

if [ -f "$REPO_DIR/.env.eval" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.eval"
  set +a
fi
export VAST_NO_AUTO_PROVISION=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

git pull -q origin main 2>/dev/null || true

PIN_FILE="${VAST_PIN_FILE:-$HOME/.sparkinfer_pinned_instance}"
INSTANCE_FILE="${VAST_INSTANCE_FILE:-$HOME/.sparkinfer_vast_instance}"
resolve_pin() {
  local v=""
  if [ -f "$PIN_FILE" ]; then
    v="$(tr -d '[:space:]' <"$PIN_FILE" 2>/dev/null || true)"
  fi
  if [ -z "$v" ] || [ "$v" = "0" ]; then
    v="${VAST_DEFAULT_INSTANCE:-${VAST_INSTANCE:-}}"
  fi
  printf '%s' "$v"
}
PINNED_ID="$(resolve_pin)"
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ] && [ -n "$PINNED_ID" ] && [ "$PINNED_ID" != "0" ]; then
  export VAST_INSTANCE="$PINNED_ID"
  export VAST_DEFAULT_INSTANCE="$PINNED_ID"
  printf '%s\n' "$PINNED_ID" >"$PIN_FILE"
  printf '%s\n' "$PINNED_ID" >"$INSTANCE_FILE"
fi

BOT_ARGS=(--repo "${REPO:-gittensor-ai-lab/sparkinfer}")
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ]; then
  BOT_ARGS+=(--instance "${VAST_INSTANCE:-0}")
fi

gpu_ready() {
  local key="${SSH_KEY:-$HOME/.ssh/speedy}"
  if [ "${EVAL_TRANSPORT:-vast}" = "ssh" ]; then
    local host="${EVAL_SSH_HOST:-}" port="${EVAL_SSH_PORT:-22}" user="${EVAL_SSH_USER:-root}"
    [ -n "$host" ] || return 1
    # IdentitiesOnly=yes: same rationale as run_dflash_cron.sh's gpu_ready — cron has no
    # ssh-agent, so without it a box whose authorized_keys only has SSH_KEY's public half (not
    # some agent identity) fails outright with "Permission denied".
    local err rc
    err="$(ssh -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 \
        -o StrictHostKeyChecking=accept-new -p "$port" "$user@$host" 'true' 2>&1)"
    rc=$?
    [ "$rc" -eq 0 ] || echo "gpu_ready: ssh to $user@$host:$port failed (exit=$rc): $err" >&2
    return "$rc"
  fi
  local iid="${VAST_INSTANCE:-}"
  [ -n "$iid" ] && [ "$iid" != "0" ] || return 1
  command -v vastai >/dev/null 2>&1 || return 1
  local raw st ip port
  raw="$(vastai show instance "$iid" --raw 2>/dev/null)" || return 1
  read -r st ip port < <(python3 -c "
import json, sys
d = json.loads(sys.stdin.read() or '{}')
st = d.get('actual_status') or ''
ip = d.get('public_ipaddr') or ''
ports = d.get('ports') or {}
p = ((ports.get('22/tcp') or [{}])[0] or {}).get('HostPort') or ''
print(st, ip, p)
" <<<"$raw")
  [ "$st" = "running" ] && [ -n "$ip" ] && [ -n "$port" ] || return 1
  ssh -i "$key" -o BatchMode=yes -o ConnectTimeout=10 \
      -o StrictHostKeyChecking=accept-new -p "$port" "root@$ip" 'true' 2>/dev/null
}

GPU_LABEL="${EVAL_SSH_HOST:-ssh}"
[ "${EVAL_TRANSPORT:-vast}" = "ssh" ] || GPU_LABEL="${VAST_INSTANCE:-?}"

TS="$(date -u +%FT%TZ)"
if gpu_ready; then
  echo "[$TS] sparkinfer Qwen3.8-27B bot — pinned GPU $GPU_LABEL up — full eval (AUTOMERGE=${SPARKINFER_QWEN38_AUTOMERGE:-0})"
  python3 eval/pr_qwen38_bot.py "${BOT_ARGS[@]}"
else
  echo "[$TS] sparkinfer Qwen3.8-27B bot — pinned GPU $GPU_LABEL down — labels only"
  python3 eval/pr_qwen38_bot.py "${BOT_ARGS[@]}" --labels-only
fi
