"""Launch an extra Blender window with the MCP add-on server on a given port, and wait until it answers.

    python tools/blender/launch_mcp_blender.py --port 9878 [--blend file.blend] [--blender PATH]
    python tools/blender/launch_mcp_blender.py --check 9876 9878 9879     # status of ports

Cross-platform (Windows / macOS / Linux). Finds Blender via --blender, $BLENDER_EXE, PATH or the
usual install folders; starts it detached (xvfb-run on Linux without a display) with
tools/blender/start_mcp_instance.py; polls the port with the add-on's JSON protocol
(get_scene_info) for up to 90 s. Never touches a port that is already served, and refuses 9877
(reserved on this project's machines).
"""
import argparse
import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FORBIDDEN = {9877}


def ping(port, timeout=3.0):
    try:
        with socket.create_connection(('localhost', port), timeout=timeout) as s:
            s.sendall(json.dumps({'type': 'get_scene_info', 'params': {}}).encode())
            s.settimeout(timeout)
            buf = b''
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
                try:
                    return json.loads(buf.decode()).get('status') == 'success'
                except json.JSONDecodeError:
                    continue
    except OSError:
        return False
    return False


def listening(port):
    try:
        with socket.create_connection(('localhost', port), timeout=1.0):
            return True
    except OSError:
        return False


def find_blender(explicit=None):
    cands = [explicit, os.environ.get('BLENDER_EXE'), shutil.which('blender')]
    sysname = platform.system()
    if sysname == 'Windows':
        pf = [os.environ.get('ProgramFiles', r'C:\Program Files'), os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')]
        for p in pf:
            cands += sorted(glob.glob(os.path.join(p, 'Blender Foundation', 'Blender 4.2*', 'blender.exe')), reverse=True)
            cands += sorted(glob.glob(os.path.join(p, 'Blender Foundation', 'Blender*', 'blender.exe')), reverse=True)
        cands += sorted(glob.glob(os.path.expandvars(r'%LOCALAPPDATA%\Programs\Blender Foundation\Blender*\blender.exe')), reverse=True)
    elif sysname == 'Darwin':
        cands += ['/Applications/Blender.app/Contents/MacOS/Blender'] + sorted(glob.glob('/Applications/Blender*.app/Contents/MacOS/Blender'))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int)
    ap.add_argument('--blend', default=None)
    ap.add_argument('--blender', default=None)
    ap.add_argument('--check', nargs='*', type=int)
    a = ap.parse_args()

    if a.check is not None:
        for p in a.check or [9876, 9878, 9879]:
            print(f'port {p}: ' + ('RESERVED (do not use)' if p in FORBIDDEN else 'MCP OK' if ping(p) else 'listening, no MCP reply' if listening(p) else 'free'))
        return 0
    if a.port is None:
        ap.error('--port is required')
    if a.port in FORBIDDEN:
        print(f'port {a.port} is reserved on this machine; use 9878, 9879, ...'); return 2
    if ping(a.port):
        print(f'port {a.port}: an MCP Blender is already serving it; nothing to do'); return 0
    if listening(a.port):
        print(f'port {a.port} is taken by something else; choose another port'); return 2
    exe = find_blender(a.blender)
    if not exe:
        print('Blender not found: pass --blender PATH or set BLENDER_EXE'); return 2

    cmd = [exe] + ([a.blend] if a.blend else []) + ['--python', os.path.join(HERE, 'start_mcp_instance.py'), '--', '--port', str(a.port)]
    env = dict(os.environ, BLENDER_PORT=str(a.port))
    if platform.system() == 'Linux' and not os.environ.get('DISPLAY') and shutil.which('xvfb-run'):
        cmd = ['xvfb-run', '-a', '-s', '-screen 0 1600x1000x24'] + cmd
    log = open(os.path.join(os.path.expanduser('~'), f'blender_mcp_{a.port}.log'), 'w')
    kw = {'stdout': log, 'stderr': subprocess.STDOUT, 'env': env}
    if platform.system() == 'Windows':
        kw['creationflags'] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw['start_new_session'] = True
    p = subprocess.Popen(cmd, **kw)
    print(f'launched Blender pid {p.pid} for port {a.port} (log: {log.name})')
    for _ in range(90):
        if ping(a.port):
            print(f'port {a.port}: MCP OK'); return 0
        if p.poll() is not None:
            print(f'Blender exited early (code {p.returncode}); see {log.name}'); return 1
        time.sleep(1)
    print(f'port {a.port}: no MCP reply after 90 s; see {log.name}'); return 1


if __name__ == '__main__':
    sys.exit(main())
