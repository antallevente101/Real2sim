#!/usr/bin/env python3
"""Step 8: Calibration & tuning — neutral pose calibration, smoothing, angle scaling."""

import cv2
import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_tracker.camera import Camera
from g1_tracker.pose_tracker import PoseTracker
from g1_tracker.mujoco_sim import G1Simulator
from g1_tracker.config import WINDOW_NAME, MIRROR_CAMERA

# Calibration file
CALIB_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")


class Calibrator:
    """Handles neutral pose calibration and angle mapping."""

    def __init__(self):
        self.offsets = {}
        self.scales = {}
        self.load()

    def calibrate(self, tracker, cam):
        """Countdown then capture neutral pose, average angles as offsets."""
        print("\n📐 CALIBRATION — get in position!")

        # --- 5-second countdown ---
        countdown_start = time.time()
        last_sec = 6
        while time.time() - countdown_start < 5.0:
            ok, frame = cam.read()
            if not ok:
                continue
            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            results, _ = tracker.process(frame)
            cd_frame = tracker.draw_landmarks(frame, results)
            remaining = 5 - int(time.time() - countdown_start)
            if remaining != last_sec:
                print(f"   ⏳ {remaining}...")
                last_sec = remaining
            cv2.putText(cd_frame, f"Get ready... {remaining}s",
                        (cd_frame.shape[1] // 2 - 120, cd_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
            cv2.imshow(WINDOW_NAME, cd_frame)
            cv2.waitKey(1)

        # --- 2-second capture ---
        print("   🎯 Capturing for 2 seconds — HOLD STILL!")
        samples = []
        capture_start = time.time()

        while time.time() - capture_start < 2.0:
            ok, frame = cam.read()
            if not ok:
                continue
            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            results, _ = tracker.process(frame)
            angles = tracker.extract_arm_angles(results, frame.shape)
            if angles:
                samples.append(angles)
            # Show capture progress
            elapsed = time.time() - capture_start
            progress_frame = tracker.draw_landmarks(frame, results)
            bar_w = int((elapsed / 2.0) * progress_frame.shape[1])
            cv2.rectangle(progress_frame, (0, progress_frame.shape[0] - 20),
                          (bar_w, progress_frame.shape[0]), (0, 255, 0), -1)
            cv2.putText(progress_frame, "CAPTURING — hold still!", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, progress_frame)
            cv2.waitKey(1)

        if len(samples) < 5:
            print("⚠️  Not enough samples for calibration")
            return False

        # Average across all samples (collect all keys, not just from first frame)
        avg_angles = {}
        all_keys = set()
        for s in samples:
            all_keys.update(s.keys())
        for key in all_keys:
            vals = [s[key] for s in samples if key in s]
            if vals:
                avg_angles[key] = sum(vals) / len(vals)

        self.offsets = avg_angles
        self.scales = {k: 1.0 for k in avg_angles}

        self.save()
        print(f"✅ Calibrated! Offsets: {', '.join(f'{k}={np.degrees(v):.1f}°' for k, v in avg_angles.items())}")
        return True

    def apply(self, angles):
        """Apply calibration offsets and scaling to raw angles."""
        if not angles or not self.offsets:
            return angles

        calibrated = {}
        for key, val in angles.items():
            if key in self.offsets:
                calibrated[key] = (val - self.offsets[key]) * self.scales.get(key, 1.0)
            else:
                calibrated[key] = val
        return calibrated

    def save(self):
        data = {
            "offsets": {k: float(v) for k, v in self.offsets.items()},
            "scales": {k: float(v) for k, v in self.scales.items()},
        }
        with open(CALIB_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"   Calibration saved to {CALIB_FILE}")

    def load(self):
        if os.path.exists(CALIB_FILE):
            with open(CALIB_FILE, "r") as f:
                data = json.load(f)
            self.offsets = data.get("offsets", {})
            self.scales = data.get("scales", {})
            print(f"📂 Loaded calibration from {CALIB_FILE}")


def create_side_by_side(cam_frame, mujoco_rgb, target_height=480):
    cam_h, cam_w = cam_frame.shape[:2]
    scale = target_height / cam_h
    cam_scaled = cv2.resize(cam_frame, (int(cam_w * scale), target_height))
    mj_bgr = cv2.cvtColor(mujoco_rgb, cv2.COLOR_RGB2BGR)
    mj_h, mj_w = mj_bgr.shape[:2]
    mj_scaled = cv2.resize(mj_bgr, (int(mj_w * target_height / mj_h), target_height))
    return np.hstack([cam_scaled, mj_scaled])


def get_screen_size():
    """Get screen resolution for fullscreen scaling."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080  # fallback


def setup_fullscreen_window(window_name):
    """Create a fullscreen OpenCV window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def ensure_fullscreen(window_name):
    """Make sure the window stays fullscreen (call after any imshow that might reset)."""
    prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
    if prop != cv2.WINDOW_FULLSCREEN:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def main():
    print("=" * 50)
    print("Step 8: Calibration & Tuning")
    print("Press 'q' to quit, 'r' to reset, 'c' to calibrate")
    print("=" * 50)

    cam = Camera().open()
    tracker = PoseTracker()
    sim = G1Simulator(width=480, height=480)
    calibrator = Calibrator()
    sim.stand_pose()

    # Fullscreen setup
    screen_w, screen_h = get_screen_size()
    setup_fullscreen_window(WINDOW_NAME)
    print(f"Fullscreen: {screen_w}x{screen_h}")

    smoothed = {}
    smoothing = 0.3  # default smoothing
    scale_factor = 1.0  # angle amplification (adjust with [ ] keys)
    frame_count = 0
    fps = 0
    fps_timer = time.time()
    render_interval = 3  # render robot every Nth frame (saves 330ms/frame)
    last_rendered = None
    freeze_frames = 5   # show neutral for first 5 frames (user gets in position)

    # Initial render so robot is visible from the start
    sim.forward()
    last_rendered = sim.render()

    try:
        while True:
            frame_count += 1
            ok, cam_frame = cam.read()
            if not ok:
                time.sleep(0.01)
                continue

            if MIRROR_CAMERA:
                cam_frame = cv2.flip(cam_frame, 1)

            results, _ = tracker.process(cam_frame)
            angles = tracker.extract_arm_angles(results, cam_frame.shape)

            # Apply calibration
            calibrated = calibrator.apply(angles)

            # Freeze: skip tracking for N frames after reset
            if freeze_frames > 0:
                freeze_frames -= 1
                calibrated = None  # don't update robot during freeze

            # Scale for more dramatic movement
            if calibrated:
                for k in calibrated:
                    calibrated[k] *= scale_factor

            # G1 elbow neutral bias — applied AFTER calibration (not captured by it)
            ELBOW_NEUTRAL = 0.3
            if calibrated:
                for side in ['left', 'right']:
                    key = f'{side}_elbow'
                    if key in calibrated:
                        calibrated[key] += ELBOW_NEUTRAL

            # Smooth
            if calibrated:
                for key, val in calibrated.items():
                    old = smoothed.get(key, val)
                    smoothed[key] = smoothing * val + (1 - smoothing) * old

            # MuJoCo — forward kinematics every frame (fast), render every Nth (slow)
            if smoothed:
                sim.set_joint_angles(smoothed)
            sim.forward()
            if frame_count % render_interval == 0:
                last_rendered = sim.render()

            # Display
            cam_display = tracker.draw_landmarks(cam_frame, results)
            status = "TRACKING" if angles else "No person"
            status_color = (0, 255, 0) if angles else (0, 0, 255)
            cv2.putText(cam_display, f"Camera — {status}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Show calibrated angles
            y = 55
            if smoothed:
                cv2.rectangle(cam_display, (5, 40), (300, 180), (30, 30, 30), -1)
                cv2.putText(cam_display, "CALIBRATED ANGLES", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
                for joint in ["left_shoulder_pitch", "left_elbow",
                              "right_shoulder_pitch", "right_elbow"]:
                    if joint in smoothed:
                        deg = np.degrees(smoothed[joint])
                        label = joint.replace("_", " ").title()
                        cv2.putText(cam_display, f"{label}: {deg:+.1f}°", (15, y + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        y += 20

            # Side-by-side, scaled to fullscreen
            combined = create_side_by_side(cam_display, last_rendered, target_height=screen_h)
            # If too wide, letterbox
            if combined.shape[1] > screen_w:
                scale = screen_w / combined.shape[1]
                new_h = int(combined.shape[0] * scale)
                combined = cv2.resize(combined, (screen_w, new_h))
                # Center vertically with black bars
                if new_h < screen_h:
                    pad_top = (screen_h - new_h) // 2
                    pad_bot = screen_h - new_h - pad_top
                    combined = cv2.copyMakeBorder(combined, pad_top, pad_bot, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # FPS
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - fps_timer) if now > fps_timer else 0
                fps_timer = now

            # Render indicator — shows when robot render actually updated
            render_fresh = (frame_count % render_interval == 0)
            r_color = (0, 255, 0) if render_fresh else (100, 100, 100)
            cv2.putText(combined, f"FPS: {fps:.1f} | Smooth: {smoothing:.2f} | Robot: {'LIVE' if render_fresh else 'cached'}",
                        (10, combined.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_color, 1)
            cv2.putText(combined, "q:quit  r:reset  c:calibrate  +/-:smooth  [/]:scale",
                        (10, combined.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✅ Quit.")
                break
            elif key == ord('r'):
                sim.stand_pose()
                last_rendered = sim.render()
                smoothed.clear()
                freeze_frames = 5  # show neutral for 5 frames while user settles
                print("↺ Reset — hold still...")
            elif key == ord('c'):
                calibrator.calibrate(tracker, cam)
                smoothed.clear()
                sim.stand_pose()
                last_rendered = sim.render()
                freeze_frames = 5
                ensure_fullscreen(WINDOW_NAME)
                print("↺ Robot reset to neutral — tracking with calibration")
            elif key == ord('+') or key == ord('='):
                smoothing = min(0.95, smoothing + 0.05)
                print(f"Smoothing: {smoothing:.2f}")
            elif key == ord('-'):
                smoothing = max(0.05, smoothing - 0.05)
                print(f"Smoothing: {smoothing:.2f}")
            elif key == ord('['):
                scale_factor = max(0.1, scale_factor - 0.1)
                print(f"Scale: {scale_factor:.1f}")
            elif key == ord(']'):
                scale_factor = min(5.0, scale_factor + 0.1)
                print(f"Scale: {scale_factor:.1f}")
    finally:
        tracker.close()
        cam.release()
        sim.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
