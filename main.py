#!/usr/bin/env python3
"""
G1 Robot Arm Tracker — Main Entry Point

Usage:
    python g1_tracker/main.py [--step 1-9] [--camera 0|1]

Steps:
    1 — Raw camera feed
    2 — Pose skeleton overlay
    3 — Arm joint angle extraction
    4 — MuJoCo G1 model viewer
    5 — Manual joint control (sine wave)
    6 — Camera tracking → MuJoCo integration
    7 — Side-by-side display
    8 — Calibration & tuning
    9 — Hand/wrist tracking (full)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="G1 Robot Arm Tracker — webcam arm tracking mapped to simulated G1 robot arms"
    )
    parser.add_argument("--step", type=int, choices=range(1, 10), default=7,
                        help="Which step to run (1-9, default: 7)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (0 for /dev/video0, 1 for /dev/video1)")
    args = parser.parse_args()

    # Override camera index in config
    from g1_tracker import config
    config.CAMERA_INDEX = args.camera

    # Run the selected step
    step_modules = {
        1: "g1_tracker.step1_camera_feed",
        2: "g1_tracker.step2_pose_overlay",
        3: "g1_tracker.step3_arm_angles",
        4: "g1_tracker.step4_mujoco_viewer",
        5: "g1_tracker.step5_manual_control",
        6: "g1_tracker.step6_integration",
        7: "g1_tracker.step7_side_by_side",
        8: "g1_tracker.step8_calibration",
        9: "g1_tracker.step9_wrist_tracking",
    }

    module_name = step_modules[args.step]
    print(f"\n🚀 Starting Step {args.step}: {module_name.split('.')[-1]}\n")

    import importlib
    mod = importlib.import_module(module_name)
    mod.main()


if __name__ == "__main__":
    main()
