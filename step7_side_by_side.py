#!/usr/bin/env python3
"""Step 7: Side-by-side display — camera feed + MuJoCo render in one window."""

import cv2
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_tracker.camera import Camera
from g1_tracker.pose_tracker import PoseTracker
from g1_tracker.mujoco_sim import G1Simulator
from g1_tracker.config import WINDOW_NAME, MIRROR_CAMERA, SMOOTHING_ALPHA


def create_side_by_side(cam_frame, mujoco_rgb, target_height=480):
    """Combine camera frame and MuJoCo render into one side-by-side image."""
    cam_h, cam_w = cam_frame.shape[:2]

    # Scale camera frame to target height, maintain aspect ratio
    scale = target_height / cam_h
    cam_scaled = cv2.resize(cam_frame, (int(cam_w * scale), target_height))

    # MuJoCo render is already RGB, convert to BGR
    mj_bgr = cv2.cvtColor(mujoco_rgb, cv2.COLOR_RGB2BGR)
    mj_h, mj_w = mj_bgr.shape[:2]

    # Scale MuJoCo to same height
    mj_scaled = cv2.resize(mj_bgr, (int(mj_w * target_height / mj_h), target_height))

    # Combine
    combined = np.hstack([cam_scaled, mj_scaled])
    return combined


def main():
    print("=" * 50)
    print("Step 7: Side-by-Side Display")
    print("Left: Camera + skeleton | Right: G1 Robot")
    print("Press 'q' to quit, 'r' to reset robot, 'c' to calibrate")
    print("=" * 50)

    cam = Camera().open()
    tracker = PoseTracker()
    sim = G1Simulator(width=640, height=480)
    sim.stand_pose()

    smoothed = {}
    frame_count = 0
    fps = 0
    fps_timer = time.time()

    try:
        while True:
            # Camera
            ok, cam_frame = cam.read()
            if not ok:
                time.sleep(0.01)
                continue

            if MIRROR_CAMERA:
                cam_frame = cv2.flip(cam_frame, 1)

            # Pose
            results, _ = tracker.process(cam_frame)
            angles = tracker.extract_arm_angles(results, cam_frame.shape)

            # Smooth
            if angles:
                for key, val in angles.items():
                    old = smoothed.get(key, val)
                    smoothed[key] = SMOOTHING_ALPHA * val + (1 - SMOOTHING_ALPHA) * old

            # Update MuJoCo
            if smoothed:
                sim.set_joint_angles(smoothed)
            sim.forward()
            if frame_count % 3 == 0:
                rendered = sim.render()

            # Annotate camera frame
            cam_display = tracker.draw_landmarks(cam_frame, results)
            status = "TRACKING" if angles else "No person"
            status_color = (0, 255, 0) if angles else (0, 0, 255)
            cv2.putText(cam_display, f"Camera — {status}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # FPS
            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - fps_timer) if now > fps_timer else 0
                fps_timer = now

            # Annotate MuJoCo frame
            mj_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            cv2.putText(mj_bgr, "G1 Robot", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Side-by-side
            combined = create_side_by_side(cam_display, rendered)

            # FPS overlay on combined
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, combined.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(combined, "q:quit  r:reset  c:calibrate", (combined.shape[1] - 300, combined.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✅ Quit.")
                break
            elif key == ord('r'):
                sim.stand_pose()
                smoothed.clear()
                print("↺ Robot reset")
            elif key == ord('c'):
                print("📐 Calibration: use --step 8 for full calibration")
    finally:
        tracker.close()
        cam.release()
        sim.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
