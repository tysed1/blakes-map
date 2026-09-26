#!/usr/bin/env bash
# One-time setup of headless Blender + the Blender MCP bridge.
#
#   tools/blender/setup.sh            # installs into /opt/blender (Linux x64)
#
# Installs:
#   * Blender 4.2 LTS (portable tarball)       -> /opt/blender, symlinked to /usr/local/bin/blender
#   * Xvfb + GL libs (virtual display; the MCP add-on's socket server needs
#     Blender's event loop, which does not run in `blender -b` background mode)
#   * blender-mcp MCP server (uv tool)         -> `blender-mcp` / `uvx blender-mcp`
#   * the matching Blender add-on (bundled with the blender-mcp package)
set -euo pipefail
BLENDER_VERSION="${BLENDER_VERSION:-4.2.3}"
BLENDER_SERIES="${BLENDER_VERSION%.*}"
PREFIX="${BLENDER_PREFIX:-/opt/blender}"
BLENDER_DIR="$PREFIX/blender-${BLENDER_VERSION}-linux-x64"

if [ ! -x "$BLENDER_DIR/blender" ]; then
  mkdir -p "$PREFIX"
  curl -fL --retry 4 -o "$PREFIX/blender.tar.xz" \
    "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  tar -xf "$PREFIX/blender.tar.xz" -C "$PREFIX"
  rm -f "$PREFIX/blender.tar.xz"
fi
ln -sf "$BLENDER_DIR/blender" /usr/local/bin/blender

if ! command -v Xvfb >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq xvfb xauth libxi6 libxxf86vm1 libxfixes3 \
    libxrender1 libgl1 libegl1 libxkbcommon0 libsm6 libice6 libglu1-mesa libgl1-mesa-dri
fi

if ! command -v uv >/dev/null 2>&1; then
  pip3 install --user uv
fi
export PATH="$HOME/.local/bin:$PATH"
uv tool install --upgrade blender-mcp

# Install the add-on version that ships with (and matches) the installed server.
ADDONS_DIR="$HOME/.config/blender/${BLENDER_SERIES}/scripts/addons"
mkdir -p "$ADDONS_DIR"
BUNDLED="$(uv tool dir)/blender-mcp/lib/python3*/site-packages/blender_mcp/bundled/addon.py"
cp $BUNDLED "$ADDONS_DIR/blender_mcp.py"

blender -b --version | head -1
echo "Blender MCP add-on installed to $ADDONS_DIR/blender_mcp.py"
echo "Start the bridge with: tools/blender/start_mcp.sh"
