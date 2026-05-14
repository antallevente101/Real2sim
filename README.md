# Real2sim

Real-time webcam arm tracking mapped to a simulated Unitree G1 humanoid robot in MuJoCo.

Point your webcam at yourself, and the G1 robot mirrors your arm movements in real time. MediaPipe Pose extracts your skeleton, Direct Joint Retargeting decomposes arm vectors into 14 joint angles using a torso reference frame, and MuJoCo renders the robot — all running on WSL2 with a TCP camera bridge to Windows.

## How it works

Windows webcam → TCP bridge → WSL2 → MediaPipe Pose → torso-frame angle decomposition → 14 G1 joint angles → MuJoCo renderer

## Features

- Fullscreen side-by-side display (camera + skeleton | G1 robot)
- One-press calibration: stand still for 2s, offsets saved automatically
- Adjustable smoothing and angle scaling on the fly
- Camera server auto-launches on Windows — no manual CMD needed
- Render throttling keeps the camera feed responsive despite ~330ms STL rasterization

## Requirements

- Windows 11 with WSL2 (WSLg)
- Python 3.10+ on both sides
- USB webcam
- MuJoCo, MediaPipe, OpenCV, NumPy, SciPy
