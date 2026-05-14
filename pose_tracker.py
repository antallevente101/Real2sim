"""Pose tracking with MediaPipe — extracts arm joint angles.

Uses MediaPipe 0.10.x+ task API (vision.PoseLandmarker).
"""

import os
import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe import Image as MPImage, ImageFormat
from g1_tracker.config import (
    POSE_MODEL_COMPLEXITY,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
)

# Path to the pose landmarker model
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_POSE_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker.task")

# Landmark indices (same as old MediaPipe)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0


class PoseTracker:
    """Wraps MediaPipe PoseLandmarker for skeleton overlay and arm-angle extraction."""

    # Connection pairs for drawing (landmark index pairs)
    CONNECTIONS = [
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 12),             # shoulders
        (11, 23), (12, 24),   # torso sides
        (23, 24),             # hips
        (0, 11), (0, 12),     # nose to shoulders
    ]

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = _POSE_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose landmarker model not found: {model_path}\n"
                f"Download from: https://storage.googleapis.com/mediapipe-models/"
                f"pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            )

        base_opts = base_options_module.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=POSE_MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        """Run MediaPipe on a BGR frame. Returns (landmarks_list, None).

        landmarks_list is a list of lists of NormalizedLandmark, or [] if no person.
        Compatibility: second return is always None (old API compat).
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        if result.pose_landmarks:
            return result.pose_landmarks, None
        return [], None

    def draw_landmarks(self, frame_bgr, landmarks_list, draw_numbers=False):
        """Draw pose skeleton on the frame. Returns annotated frame."""
        if not landmarks_list:
            return frame_bgr

        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        for landmarks in landmarks_list:
            # Draw connections
            for start_idx, end_idx in self.CONNECTIONS:
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]

                # Skip low-confidence landmarks
                if getattr(start_lm, 'visibility', 1.0) < 0.5:
                    continue
                if getattr(end_lm, 'visibility', 1.0) < 0.5:
                    continue

                pt1 = (int(start_lm.x * w), int(start_lm.y * h))
                pt2 = (int(end_lm.x * w), int(end_lm.y * h))

                cv2.line(annotated, pt1, pt2, (0, 255, 100), 2)

            # Draw landmark points
            for i, lm in enumerate(landmarks):
                vis = getattr(lm, 'visibility', 1.0)
                if vis < 0.5:
                    continue
                cx, cy = int(lm.x * w), int(lm.y * h)
                radius = 6 if i in (11, 12, 13, 14, 15, 16) else 3
                color = (0, 200, 255)
                cv2.circle(annotated, (cx, cy), radius, color, -1)

                if draw_numbers:
                    cv2.putText(annotated, str(i), (cx + 8, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return annotated

    def extract_arm_angles(self, landmarks_list, frame_shape):
        """
        Direct Joint Retargeting: build torso reference frame, express arm
        vectors in torso-local coords, decompose to G1 joint angles.
        Returns dict with 14 keys (7 per arm). All in radians.
        """
        if not landmarks_list:
            return None

        lm = landmarks_list[0]

        def lm3d(idx):
            l = lm[idx]
            return np.array([l.x, l.y, l.z])

        # --- Build torso reference frame ---
        # X = forward (toward camera), Y = right (shoulder line), Z = up (spine)
        left_sh  = lm3d(LEFT_SHOULDER)
        right_sh = lm3d(RIGHT_SHOULDER)
        left_hip  = lm3d(LEFT_HIP)
        right_hip = lm3d(RIGHT_HIP)
        mid_sh = (left_sh + right_sh) / 2
        mid_hip = (left_hip + right_hip) / 2

        Y = right_sh - left_sh
        Y = Y / (np.linalg.norm(Y) + 1e-8)
        Z_rough = mid_sh - mid_hip  # spine direction (roughly upward)
        X = np.cross(Y, Z_rough)
        X = X / (np.linalg.norm(X) + 1e-8)
        Z = np.cross(X, Y)  # true upward (perpendicular to shoulders + forward)

        def torso_frame(v):
            """Express world vector in torso-local frame."""
            return np.array([np.dot(v, X), np.dot(v, Y), np.dot(v, Z)])

        angles = {}

        for side, sh_idx, el_idx, wr_idx in [
            ("left", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            ("right", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        ]:
            if any(getattr(lm[i], 'visibility', 1.0) < 0.5
                   for i in [sh_idx, el_idx]):
                continue

            sh = lm3d(sh_idx)
            el = lm3d(el_idx)
            wr = lm3d(wr_idx) if getattr(lm[wr_idx], 'visibility', 1.0) >= 0.5 else None

            # Upper arm direction in torso frame
            ua_w = el - sh
            ua_t = torso_frame(ua_w)
            ua_t_norm = ua_t / (np.linalg.norm(ua_t) + 1e-8)

            # --- Shoulder pitch: rotation around Y (forward/backward) ---
            # arctan2(x, -z): 0 when arm down (-z), positive when arm forward (+x)
            pitch = np.arctan2(ua_t_norm[0], -ua_t_norm[2])
            angles[f"{side}_shoulder_pitch"] = np.clip(pitch, -2.5, 2.5)

            # --- Shoulder roll: rotation around X (side/abduction) ---
            roll = np.arctan2(ua_t_norm[1], -ua_t_norm[2])
            if side == "left":
                # Left G1: positive roll = abduction (arm to left = -Y)
                # roll negative when arm to left → negate for positive abduction
                angles[f"{side}_shoulder_roll"] = np.clip(-roll, -1.5, 2.2)
            else:
                # Right G1: negative roll = abduction (arm to right = +Y)
                # roll positive when arm to right → negate for negative abduction
                angles[f"{side}_shoulder_roll"] = np.clip(-roll, -2.2, 1.5)

            # --- Shoulder yaw: rotation of elbow-bend plane around upper arm ---
            if wr is not None:
                fa_w = wr - el
                fa_t = torso_frame(fa_w)
                # Project forearm onto plane perpendicular to upper arm
                ua_dir_t = ua_t_norm
                fa_perp = fa_t - np.dot(fa_t, ua_dir_t) * ua_dir_t
                fa_perp = fa_perp / (np.linalg.norm(fa_perp) + 1e-8)
                # Reference elbow-bend direction in torso frame: roughly forward (+X)
                ref = np.array([1.0, 0.0, 0.0])
                ref_perp = ref - np.dot(ref, ua_dir_t) * ua_dir_t
                ref_perp = ref_perp / (np.linalg.norm(ref_perp) + 1e-8)
                # Signed angle between ref and fa in the perpendicular plane
                cross_axis = np.cross(ref_perp, fa_perp)
                yaw = np.arcsin(np.clip(np.dot(cross_axis, ua_dir_t), -1, 1))
                angles[f"{side}_shoulder_yaw"] = np.clip(yaw, -1.0, 1.0)
            else:
                angles[f"{side}_shoulder_yaw"] = 0.0

            # --- Elbow: 2D image-plane angle (ignores noisy z-depth) ---
            if wr is not None:
                ua_2d = np.array([el[0] - sh[0], el[1] - sh[1]])
                fa_2d = np.array([wr[0] - el[0], wr[1] - el[1]])
                dot_2d = np.dot(ua_2d, fa_2d)
                norm_prod = np.linalg.norm(ua_2d) * np.linalg.norm(fa_2d) + 1e-8
                # Raw elbow angle (no G1 bias — bias added after calibration)
                elbow_2d = np.arccos(np.clip(dot_2d / norm_prod, -1, 1))
                angles[f"{side}_elbow"] = elbow_2d

            # --- Wrist: forearm direction in torso frame ---
            if wr is not None:
                fa_t_dir = torso_frame(wr - el)
                fa_t_dir = fa_t_dir / (np.linalg.norm(fa_t_dir) + 1e-8)
                # Pitch: forward/backward tilt (from straight-down = 0)
                wrist_pitch = np.arctan2(fa_t_dir[0], -fa_t_dir[2]) * 0.5
                # Yaw: left/right deviation
                wrist_yaw = np.arctan2(fa_t_dir[1], -fa_t_dir[2]) * 0.5
                angles[f"{side}_wrist_pitch"] = np.clip(wrist_pitch, -1.5, 1.5)
                angles[f"{side}_wrist_yaw"] = np.clip(wrist_yaw, -1.0, 1.0)
            angles[f"{side}_wrist_roll"] = 0.0  # needs hand landmarks

        return angles

    def extract_wrist_angles(self, landmarks_list, frame_shape):
        """Extract approximate wrist angles from pose landmarks (fallback)."""
        if not landmarks_list:
            return None

        h, w = frame_shape[:2]
        landmarks = landmarks_list[0]
        angles = {}

        for side, el, wr in [
            ("left", LEFT_ELBOW, LEFT_WRIST),
            ("right", RIGHT_ELBOW, RIGHT_WRIST),
        ]:
            el_vis = getattr(landmarks[el], 'visibility', 1.0)
            wr_vis = getattr(landmarks[wr], 'visibility', 1.0)
            if el_vis < 0.5 or wr_vis < 0.5:
                continue

            forearm = np.array([landmarks[wr].x - landmarks[el].x,
                                landmarks[wr].y - landmarks[el].y])
            wrist_pitch = np.arctan2(forearm[1], np.abs(forearm[0]) + 1e-8)
            wrist_yaw = np.arctan2(forearm[0], np.abs(forearm[1]) + 1e-8) * 0.5

            angles[f"{side}_wrist_pitch"] = wrist_pitch
            angles[f"{side}_wrist_yaw"] = wrist_yaw
            angles[f"{side}_wrist_roll"] = 0.0

        return angles

    def close(self):
        self.detector.close()
