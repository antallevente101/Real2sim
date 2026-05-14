# Real2sim — Complete Reconstruction Guide

> **FINAL VERSION** — May 2026
> Real-time webcam arm tracking mapped to a simulated Unitree G1 humanoid robot in MuJoCo.
> Reconstruct this project from scratch, including every fix discovered during development.

---

## Required Software

### Windows (host machine)
| Software | How to install |
|----------|---------------|
| Python 3.10+ | https://python.org or `winget install python` |
| OpenCV (Python) | `pip install opencv-python` |
| USB webcam | Built-in or external |

### WSL2 Ubuntu
| Software | How to install |
|----------|---------------|
| Python 3 | Comes with Ubuntu, or `sudo apt install python3` |
| pip | `sudo apt install python3-pip` |
| MuJoCo | `pip install mujoco` |
| OpenCV (Python) | `pip install opencv-python` |
| MediaPipe | `pip install mediapipe` |
| NumPy | `pip install numpy` |
| SciPy | `pip install scipy` |
| Unitree G1 model | `git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git` |
| Mesh symlink | `ln -sf unitree_mujoco/unitree_robots/g1/meshes meshes` |

### Verify WSLg GPU passthrough
```bash
ls /dev/dxg  # must exist for GPU rendering
```

---

## What It Is

```
Windows camera_server.py ──TCP:9999──► WSL2 → MediaPipe Pose → 14 joint angles → MuJoCo G1
```

**Platform:** WSL2 Ubuntu on Windows 11 · Python 3 · EGL rendering via WSLg

---

## Project Structure

```
real2sim/
├── camera_server.py          # Windows-side camera capture (TCP server)
├── g1_scene.xml              # MuJoCo scene: G1 robot + cameras + light
├── g1_scene_fast.xml         # Simplified G1 model (primitives, not STL meshes)
├── RECONSTRUCTION.md         # This guide
└── g1_tracker/
    ├── config.py             # All tunable parameters
    ├── camera.py             # WSL-side TCP camera client (auto-launches Windows server)
    ├── pose_tracker.py       # MediaPipe pose → Direct Joint Retargeting → G1 angles
    ├── mujoco_sim.py         # MuJoCo G1 model loader + offscreen renderer
    ├── main.py               # Unified launcher: --step 1-9
    ├── step7_side_by_side.py # Side-by-side camera + robot display
    ├── step8_calibration.py  # Calibration with countdown + freeze + bias (fullscreen)
    ├── setup_env.sh          # Environment variables
    ├── pose_landmarker.task  # MediaPipe model (30 MB, auto-downloaded)
    ├── hand_landmarker.task  # MediaPipe hand model (unused, for Step 9)
    └── __init__.py
```

---

## Step 1: Scene File

**File: `g1_scene.xml`**

Key details that took hours to get right:

- **Single light** at `pos="4 4 4"` (top-right corner). Multiple lights caused double shadows.
- **Front camera** `xyaxes="0 1 0 0 0 1"` — cross product X×Y=Z=(1,0,0), -Z=(-1,0,0) looks toward robot. The original `"0 -1 0"` pointed AWAY from the robot!
- **No `<compiler>` tag** — it's in the included G1 model file and would conflict if duplicated.
- **Programmatic headlight boost** in `mujoco_sim.py` — XML headlight values are baseline only.

---

## Step 2: Camera Bridge

### The fundamental problem
WSL2 cannot directly access USB webcams. USB/IP uses bulk transfer; UVC webcams need isochronous transfer. The camera enumerates as `/dev/video0` but delivers zero frames.

### Solution: TCP bridge
- **Windows:** `camera_server.py` captures via DirectShow (MJPG, 640×480), JPEG-encodes at quality 50, sends over TCP with 4-byte length prefix
- **WSL:** `g1_tracker/camera.py` connects to Windows host IP (auto-resolved via `ip route`), receives JPEG frames, decodes to numpy

### Critical notes
- **`127.0.0.1` does not work** — WSL2 has separate network namespace. Use `172.22.176.1` (WSL gateway).
- **`TCP_NODELAY`** on the server socket — disables Nagle's algorithm (40ms buffer delay)
- **JPEG quality 50** — balances size (~15KB/frame) and quality (enough for MediaPipe)
- **No double-flip** — `MIRROR_CAMERA=True` in config handles the mirror on WSL side only
- **Auto-launch** — if connection fails, `camera.py` runs `cmd.exe /c start "" python camera_server.py` on Windows automatically

### Latency
Rendering is ~330ms/frame on WSL2 (G1 model has 60+ STL meshes). To maintain responsiveness:
- **Render every 3rd frame** — forward kinematics runs every frame, only rendering is throttled
- Camera feed + skeleton overlay updates at full speed
- Robot render indicator shows `LIVE` (green) or `cached` (gray)

---

## Step 3: Direct Joint Retargeting (arm angle extraction)

### The approach
Instead of computing angles from flat 2D projections or unreliable 3D depth, we build a **torso reference frame** from the shoulders and hips, express arm vectors in torso-local coordinates, and decompose into G1 joint angles.

### Torso frame
```
X = forward (toward camera)    = cross(Y, Z_rough)
Y = right (along shoulder line) = right_shoulder - left_shoulder
Z = up (along spine)            = cross(X, Y)
```

### Angle decomposition

**Shoulder pitch** (forward/backward, rotation around Y):
```python
pitch = arctan2(ua_t_x, -ua_t_z)  # 0 when arm down, positive when forward
```

**Shoulder roll** (abduction, rotation around X):
```python
roll = arctan2(ua_t_y, -ua_t_z)
# Left arm:  negate (G1 positive = abduction)
# Right arm: negate (G1 negative = abduction, mirrored from left)
```

**Shoulder yaw** (rotation of elbow-bend plane around upper arm axis):
```python
# Project forearm onto plane perpendicular to upper arm
# Compute signed angle between reference (forward) and actual elbow-bend direction
```

**Elbow** (angle between upper arm and forearm):
- Uses **2D pixel coordinates only** (x,y from image) — avoids MediaPipe's noisy z-depth
- G1 model has a **permanent slight bend at qpos=0**. We add +0.3 rad bias AFTER calibration so it's not captured by calibration offsets.

**Wrist** (forearm direction in torso frame):
- Pitch: forward/backward deviation from straight-down
- Yaw: left/right deviation
- Roll: always 0 (needs hand landmarks from Step 9 for this)

### Why we use 2D for elbow
MediaPipe's z-depth is approximate from a single camera. When arms are at sides, the wrist is often estimated 10-20cm forward of its true position, making a straight arm look bent in 3D. Pixel positions (x,y) are accurate.

---

## Step 4: MuJoCo Rendering

### Backend
- **EGL:** `MUJOCO_GL=egl` — avoids GLX BadAccess crash on WSLg. GLFW internally uses GLX which is broken on WSL2/WSLg.
- **Headlight:** Programmatic boost `ambient=[0.8,0.8,0.8]`, `diffuse=[1,1,1]`

### Render throttling
Every 3rd frame only. `forward()` runs every frame for kinematics.

### G1 model quirk
The G1 elbow at qpos=0 is mechanically bent (~17° forward lean). The straightest-looking angle is **+0.3 rad**. This bias is applied in `step8_calibration.py`, not in `pose_tracker.py`, so calibration doesn't capture it.

---

## Step 5: Calibration

### How it works
1. Press `c` → 5-second countdown on screen (gives time to walk to position)
2. 2-second capture (green progress bar, must hold still)
3. Angles averaged across all samples → saved to `g1_tracker/calibration.json`
4. Robot resets to neutral, freezes for 5 frames, then tracks with calibration applied
5. Window stays fullscreen throughout

### Critical fix: key averaging bug
The original code only averaged keys from the **first** captured sample. If the right arm wasn't detected in the first frame, it was permanently ignored. Fixed by collecting all keys from all samples:
```python
all_keys = set()
for s in samples:
    all_keys.update(s.keys())
```

### Critical fix: elbow bias placement
The G1 elbow bias (+0.3 rad) must be applied **after** calibration subtraction. If it's in the raw angle, calibration captures it, then subtracts it → elbow goes to 0 (bent).

```
Raw angle (pose_tracker):  raw_elbow  (no bias)
Calibration offset:         raw_elbow  (captured from neutral pose)
After calibration:          raw - offset = 0
After bias (step8):         0 + 0.3 = 0.3  → straight! ✓
```

### Freeze period
After startup, reset ('r'), and calibration ('c'), tracking is frozen for 5 frames so the robot shows its neutral pose while you settle into position.

---

## Step 6: Running the Program

### Every session
```bash
cd ~/real2sim
source g1_tracker/setup_env.sh
python3 g1_tracker/main.py --step 8

# The camera server auto-launches on Windows. No manual CMD needed.
# Window opens in fullscreen. Content scales to screen height.
```

### Controls
| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset robot to neutral (5-frame freeze) |
| `c` | Calibrate (5s countdown → 2s capture → freeze, stays fullscreen) |
| `+`/`-` | Increase/decrease smoothing |
| `[`/`]` | Decrease/increase scale factor |

### After calibration
- Delete `g1_tracker/calibration.json` to reset calibration
- Stand with arms **completely straight at sides** during capture
- After calibration, robot stays at neutral (straight-looking arms)

---

## All Pitfalls Fixed (chronological order)

| # | Problem | Symptom | Root Cause | Fix |
|---|---------|---------|------------|-----|
| 1 | Camera doesn't work in WSL2 | `/dev/video0` times out, zero frames | USB/IP bulk transfer ≠ UVC isochronous | TCP bridge: capture on Windows, send JPEG over TCP |
| 2 | WSL can't reach Windows | `ConnectionRefusedError` on `127.0.0.1` | Separate network namespaces | Use WSL gateway IP (`172.22.176.1`) |
| 3 | Windows Firewall blocks | `ConnectionRefusedError` with correct IP | Defender blocks Python first run | Click "Allow" on popup, or add Python in Firewall settings |
| 4 | `camera_server.py` SyntaxError | `\U` in docstring interpreted as Unicode escape | Python string escaping | Raw string docstring: `r"""..."""` |
| 5 | Auto-launch fails on Windows | `start "G1 Camera"` treats title as command | Windows `start` syntax | `start "" python "path"` (empty title) |
| 6 | MuJoCo renders pitch black | Render maxes at 13/255 | EGL/llvmpipe software rendering | Programmatic headlight boost + front spotlight |
| 7 | Mesh files not found | `ValueError: Error opening file 'meshes/...'` | meshdir resolves relative to parent XML | `ln -sf unitree_mujoco/.../meshes meshes/` |
| 8 | GL context caches old size | Black frames after resize | EGL context not recreated | Free + recreate on every resize |
| 9 | Front camera shows empty scene | No robot visible, only floor/sky | `xyaxes="0 -1 0"` pointed AWAY from robot | Correct to `xyaxes="0 1 0 0 0 1"` (X×Y=Z toward robot) |
| 10 | Double shadows on floor | Two distinct shadows | Multiple light sources in scene | Single light at `pos="4 4 4"` (top-right) |
| 11 | Latency ~3fps (everything slow) | 330ms render blocks camera+tracking | 60+ STL meshes software-rasterized | Render every 3rd frame, forward every frame |
| 12 | 3D angles cause bent elbows at rest | Elbow computed as 55° when arm is straight | MediaPipe z-depth noise makes forearm vector wrong | Switch to 2D pixel coords for elbow |
| 13 | Arm forward → robot arm goes sideways | Shoulder roll captures vertical movement | 3D shoulder roll formula misinterprets arm-up | Direct Joint Retargeting with torso frame |
| 14 | Right arm goes opposite direction | Right roll sign flipped | G1 right shoulder roll uses mirrored convention | Negate right arm roll |
| 15 | Arm goes through robot's head | Shoulder pitch exceeds 180° | Can't distinguish "arm up" from "arm forward" in 2D | Cap pitch at 90° maximum |
| 16 | Calibration only captures one arm | Right arm not in first sample | Loop only used keys from first frame | Collect all keys from all samples |
| 17 | Robot starts with bent elbows | Elbows bent at startup | G1 model at qpos=0 has ~17° natural bend | Add +0.3 rad elbow bias in stand_pose and tracking |
| 18 | Calibration erases elbow bias | Arms go straight → pop back to bent | Bias was in raw angle, captured by calibration, then subtracted away | Move bias AFTER calibration in step8 |
| 19 | Tracking kicks in before user is ready | Robot bends before user settles | First frame captures non-neutral pose | 5-frame freeze after startup/reset/calibration |
| 20 | GLX BadAccess crash | `X Error: BadAccess (GLX)` | GLFW internally uses GLX which is broken on WSLg | Switch to `MUJOCO_GL=egl` |

---

## Environment Variables

| Variable | Value | Why |
|----------|-------|-----|
| `DISPLAY` | `:0` | X11 display for OpenCV windows (WSLg) |
| `MUJOCO_GL` | `egl` | EGL rendering (avoids GLX BadAccess crash on WSLg) |
| `LD_LIBRARY_PATH` | `...homebrew/lib` | Qt xcb plugin needs libSM, libICE |
| `QT_LOGGING_RULES` | `qt.qpa.fonts=false` | Suppresses noisy Qt font warnings |

---

## Joint Mapping (MediaPipe Pose → G1 MuJoCo)

```
MediaPipe angle          →  MuJoCo joint name
────────────────────────────────────────────
left_shoulder_pitch     →  left_shoulder_pitch_joint   (axis: 0 1 0)
left_shoulder_roll      →  left_shoulder_roll_joint    (axis: 1 0 0)
left_shoulder_yaw       →  left_shoulder_yaw_joint     (axis: 0 0 1)
left_elbow             →  left_elbow_joint            (axis: 0 1 0)
left_wrist_roll        →  left_wrist_roll_joint        (axis: 1 0 0)
left_wrist_pitch       →  left_wrist_pitch_joint       (axis: 0 1 0)
left_wrist_yaw         →  left_wrist_yaw_joint         (axis: 0 0 1)
... mirrored for right_*
```

---

## G1 Joint Ranges (radians)

| Joint | Range |
|-------|-------|
| shoulder_pitch | [-3.09, 2.67] |
| shoulder_roll (left) | [-1.59, 2.25] |
| shoulder_roll (right) | [-2.25, 1.59] — mirrored |
| shoulder_yaw | unlimited |
| elbow | [-1.05, 2.09] |
| wrist_pitch | unlimited |
| wrist_yaw | unlimited |

---

## Known Limitations (by design)

- **Single camera** — can't distinguish "arm up" from "arm forward" past 90°. Shoulder pitch capped.
- **Wrist roll = 0** — needs MediaPipe Hands (Step 9, code exists but not wired)
- **G1 elbow never perfectly straight** — mechanical design has ~17° permanent bend
- **~3fps robot render** — software rasterization of 60+ STL meshes. Camera feed smooth.
- **CPU/GPU both ~330ms/render** — WSL2's translation layer limits GPU benefit

---

## What NOT to change (things we broke and reverted)

- **Frame draining for latency** — corrupts TCP stream when switching blocking/non-blocking
- **Rotating robot via body wrapper** — breaks XML compiler (duplicate `<compiler>`)
- **Removing elbow bias** — G1 at qpos=0 is bent, needs +0.3 rad
- **3D depth-only angle computation** — MediaPipe z is too noisy for reliable elbow/wrist
- **Removing render throttling** — renders every frame blocks the camera loop with 330ms per frame
- **IK-based arm tracking** — tried, didn't work well. Direct Joint Retargeting is the working approach.

---

## History

- **IK version** (`g1_arm_tracker_with_IK`) — attempted inverse kinematics approach, didn't track well. Deleted.
- **Simplified G1 model** (`g1_scene_fast.xml`) — primitives instead of STL meshes for faster rendering. Kept as optional alternative. Not currently active.
- **Fullscreen** — step 8 opens fullscreen and stays fullscreen through calibration.
