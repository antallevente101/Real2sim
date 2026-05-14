#!/bin/bash
# Setup environment for G1 Robot Arm Tracker
# Source this file: source setup_env.sh

export DISPLAY="${DISPLAY:-:0}"

# Add homebrew libs for Qt xcb platform plugin
if [ -d "/home/linuxbrew/.linuxbrew/lib" ]; then
    export LD_LIBRARY_PATH="/home/linuxbrew/.linuxbrew/lib:${LD_LIBRARY_PATH}"
fi

# Suppress Qt font warnings
export QT_LOGGING_RULES="qt.qpa.fonts=false"

# MuJoCo rendering backend — use EGL (avoids GLX crash on WSLg)
# GLFW internally uses GLX which triggers BadAccess on WSLg
export MUJOCO_GL=egl

echo "G1 Tracker environment ready"
echo "  DISPLAY=$DISPLAY"
echo "  MUJOCO_GL=$MUJOCO_GL"
