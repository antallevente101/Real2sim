"""Shared configuration for the G1 Robot Arm Tracker project."""

import os

# Workspace root
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Camera
CAMERA_INDEX = 0  # /dev/video0 default; change to 1 for /dev/video1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# MediaPipe — speed-optimized for arm tracking
POSE_MODEL_COMPLEXITY = 0  # 0=lite(fast), 1=medium, 2=heavy(slow). Arm tracking works fine at 0
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# MuJoCo
G1_XML_PATH = os.path.join(WORKSPACE, "g1_scene.xml")

# Display
WINDOW_NAME = "G1 Robot Arm Tracker"
MIRROR_CAMERA = True  # mirror the camera feed like a selfie view

# Smoothing
SMOOTHING_ALPHA = 0.3  # low-pass filter: lower = smoother but laggier
ANGLE_LIMIT_DEG = 120  # max mapped angle in degrees
