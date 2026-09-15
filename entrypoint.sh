#!/bin/sh
# Apply the node's simulated network conditions, then run the node process.
# All knobs come from environment variables set in docker-compose.yml (from nodes.yaml).
set -e

IFACE=${NET_IFACE:-eth0}
LAT=${NET_LATENCY_MS:-0}
JIT=${NET_JITTER_MS:-0}
BW=${NET_BANDWIDTH_MBIT:-0}
LOSS=${NET_LOSS_PCT:-0}

NETEM=""
if [ "$LAT" != "0" ]; then
  NETEM="$NETEM delay ${LAT}ms"
  [ "$JIT" != "0" ] && NETEM="$NETEM ${JIT}ms distribution normal"
fi
[ "$LOSS" != "0" ] && NETEM="$NETEM loss ${LOSS}%"
[ "$BW" != "0" ]   && NETEM="$NETEM rate ${BW}mbit"

if [ -n "$NETEM" ]; then
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
  if tc qdisc add dev "$IFACE" root netem $NETEM; then
    echo "[entrypoint] $NODE_NAME netem on $IFACE:$NETEM"
  else
    echo "[entrypoint] WARNING: could not apply netem ($NETEM). Is cap_add NET_ADMIN set?" >&2
  fi
else
  echo "[entrypoint] $NODE_NAME no network shaping"
fi

exec "$@"
