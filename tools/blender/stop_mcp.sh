#!/usr/bin/env bash
PIDFILE="${BLENDER_MCP_PID:-/tmp/blender_mcp.pid}"
if [ -f "$PIDFILE" ]; then
  pkill -P "$(cat "$PIDFILE")" 2>/dev/null || true
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  rm -f "$PIDFILE"
fi
pkill -f "blender.*enable_mcp.py" 2>/dev/null || true
echo "stopped"
