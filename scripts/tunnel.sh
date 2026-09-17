#!/bin/sh
# Expose the coordinator (port 8000 by default) on a public https URL and print it.
#   sh tunnel.sh              # cloudflared quick tunnel (default): free, no account, no bandwidth cap, no uptime promise
#   TUNNEL=ngrok sh tunnel.sh # ngrok instead (free tier has a monthly transfer cap)
# Ctrl-C stops the tunnel. The URL is also written to /tmp/tunnel.url.
# Workers then join with:  dgpt-worker --coordinator https://<printed-host> --token ...
# cloudflared is the default after ngrok's transfer cap truncated a 43 MB weights download and collapsed a run (R14).
PORT="${1:-8000}"
TUNNEL="${TUNNEL:-cloudflared}"
LOG=/tmp/tunnel.log
: > "$LOG"
# Background + trap: a leaked cloudflared keeps a stale hostname alive that workers may still be pointed at.
if [ "$TUNNEL" = "ngrok" ]; then
  ngrok http "$PORT" --log=stdout --log-format=json > "$LOG" 2>&1 &
else
  cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate > "$LOG" 2>&1 &
fi
PID=$!
trap 'kill $PID 2>/dev/null' EXIT INT TERM
# Neither client announces its hostname synchronously, and a quick tunnel gets a new one every start:
# cloudflared prints it to the log, ngrok exposes it on local API 4040.
URL=""
for i in $(seq 1 40); do
  if [ "$TUNNEL" = "ngrok" ]; then
    URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c 'import sys,json; t=json.load(sys.stdin)["tunnels"]; print(next(x["public_url"] for x in t if x["public_url"].startswith("https")))' 2>/dev/null)
  else
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1)
  fi
  [ -n "$URL" ] && break
  sleep 1
done
# Fail loudly with the log path: "not installed" and "registration refused" look identical from outside.
if [ -z "$URL" ]; then echo "tunnel did not come up; see $LOG" >&2; exit 1; fi
echo "coordinator public URL: $URL"
echo "workers join with:      dgpt-worker --coordinator $URL --token <invite>"
[ "$TUNNEL" = "ngrok" ] || echo "(new trycloudflare hostnames take ~30-60 s to resolve in DNS; workers retry automatically)"
# Blocking on the tunnel ties its lifetime to this shell. Quick tunnels have died mid-campaign (R20), which
# forces every remote worker to restart against the new hostname.
echo "$URL" > /tmp/tunnel.url
wait $PID
