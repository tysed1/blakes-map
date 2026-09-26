"""Minimal client for the Blender MCP add-on socket (the same JSON protocol the
`blender-mcp` MCP server uses). Usage:
    python3 tools/blender/mcp_ping.py                 # get_scene_info
    python3 tools/blender/mcp_ping.py --code "print(1)" # execute_code
"""
import json
import os
import socket
import sys


def send(cmd, host="localhost", port=None, timeout=30):
    port = port or int(os.environ.get("BLENDER_PORT", "9876"))
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(json.dumps(cmd).encode())
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            try:
                return json.loads(buf.decode())
            except json.JSONDecodeError:
                continue
    raise RuntimeError("no response")


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    try:
        if "--code" in sys.argv:
            code = sys.argv[sys.argv.index("--code") + 1]
            r = send({"type": "execute_code", "params": {"code": code}})
        else:
            r = send({"type": "get_scene_info", "params": {}})
    except OSError as e:
        if not quiet:
            print("not reachable:", e)
        sys.exit(1)
    if not quiet:
        print(json.dumps(r, indent=2)[:4000])
    sys.exit(0 if r.get("status") == "success" else 2)
