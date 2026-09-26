#!/usr/bin/env bash
# Launch Blender (on a virtual X display when no display is available) with
# the MCP add-on's socket server listening on localhost:${BLENDER_PORT:-9876}.
# Optional first argument: a .blend file to open (e.g. exports/blender/world.blend).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="${BLENDER_MCP_LOG:-/tmp/blender_mcp.log}"
PIDFILE="${BLENDER_MCP_PID:-/tmp/blender_mcp.pid}"
export BLENDER_PORT="${BLENDER_PORT:-9876}"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Blender MCP bridge already running (pid $(cat "$PIDFILE"))"; exit 0
fi

ARGS=()
[ $# -ge 1 ] && ARGS+=("$1")
if [ -n "${DISPLAY:-}" ]; then
  nohup blender "${ARGS[@]}" --python "$HERE/enable_mcp.py" >"$LOG" 2>&1 &
else
  nohup xvfb-run -a -s "-screen 0 1600x1000x24" blender "${ARGS[@]}" --python "$HERE/enable_mcp.py" >"$LOG" 2>&1 &
fi
echo $! >"$PIDFILE"

for _ in $(seq 1 60); do
  if python3 "$HERE/mcp_ping.py" --quiet; then
    echo "Blender MCP bridge up on localhost:$BLENDER_PORT (log: $LOG)"; exit 0
  fi
  sleep 1
done
echo "Blender did not start the MCP server in time; see $LOG" >&2
exit 1
