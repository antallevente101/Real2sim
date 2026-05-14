"""MuJoCo model loading and offscreen rendering for the G1 robot."""

import os
import numpy as np
import mujoco

# Try EGL first (works on WSL2 without X11 GPU), fall back to GLFW
_gl_context = None
_CURRENT_BACKEND = None


def _init_gl_context(width, height):
    """Initialize an OpenGL context for offscreen rendering."""
    global _gl_context, _CURRENT_BACKEND

    # Recreate if dimensions changed
    if _gl_context is not None:
        try:
            _gl_context.free()
        except Exception:
            pass
        _gl_context = None
        _CURRENT_BACKEND = None

    # Try EGL first
    try:
        from mujoco.egl import GLContext as EGLContext
        _gl_context = EGLContext(width, height)
        _gl_context.make_current()
        _CURRENT_BACKEND = "EGL"
        return
    except Exception:
        pass

    # Try OSMesa
    try:
        from mujoco.osmesa import GLContext as OSMContext
        _gl_context = OSMContext(width, height)
        _gl_context.make_current()
        _CURRENT_BACKEND = "OSMesa"
        return
    except Exception:
        pass

    # Try GLFW (needs DISPLAY)
    try:
        if "DISPLAY" in os.environ:
            from mujoco.glfw import GLContext as GLFWContext
            _gl_context = GLFWContext(width, height)
            _gl_context.make_current()
            _CURRENT_BACKEND = "GLFW"
            return
    except Exception:
        pass

    raise RuntimeError(
        "Cannot initialize any OpenGL context. "
        "Install EGL (libegl1) or OSMesa (libosmesa6) or set DISPLAY for GLFW."
    )


def _get_gl_backend():
    """Return the current GL backend name."""
    return _CURRENT_BACKEND


from g1_tracker.config import G1_XML_PATH, ANGLE_LIMIT_DEG


# Maps from MediaPipe angle names to MuJoCo joint names
JOINT_MAP = {
    # Left arm (7 joints)
    "left_shoulder_pitch": "left_shoulder_pitch_joint",
    "left_shoulder_roll": "left_shoulder_roll_joint",
    "left_shoulder_yaw": "left_shoulder_yaw_joint",
    "left_elbow": "left_elbow_joint",
    "left_wrist_roll": "left_wrist_roll_joint",
    "left_wrist_pitch": "left_wrist_pitch_joint",
    "left_wrist_yaw": "left_wrist_yaw_joint",
    # Right arm (7 joints)
    "right_shoulder_pitch": "right_shoulder_pitch_joint",
    "right_shoulder_roll": "right_shoulder_roll_joint",
    "right_shoulder_yaw": "right_shoulder_yaw_joint",
    "right_elbow": "right_elbow_joint",
    "right_wrist_roll": "right_wrist_roll_joint",
    "right_wrist_pitch": "right_wrist_pitch_joint",
    "right_wrist_yaw": "right_wrist_yaw_joint",
}


class G1Simulator:
    """Loads and renders the G1 robot in MuJoCo (offscreen)."""

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # Suppress noisy warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="glfw")
        os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

        # Initialize OpenGL context for offscreen rendering
        _init_gl_context(width, height)

        # Load model
        if not os.path.exists(G1_XML_PATH):
            raise FileNotFoundError(f"G1 model not found: {G1_XML_PATH}")

        self.model = mujoco.MjModel.from_xml_path(G1_XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Boost lighting for EGL/llvmpipe software rendering
        self.model.vis.headlight.ambient = [0.8, 0.8, 0.8]
        self.model.vis.headlight.diffuse = [1.0, 1.0, 1.0]

        # Build joint name → ID lookup
        self.joint_ids = {}
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            self.joint_ids[name] = i

        # Offscreen renderer (640x480 for side-by-side)
        self.renderer = mujoco.Renderer(self.model, 480, 640)

        print(f"G1 model loaded: {self.model.njnt} joints, {self.model.nq} DoF (GL: {_CURRENT_BACKEND})")

    def set_joint_angle(self, angle_name, value_rad):
        """Set a single joint angle by its MediaPipe-style name."""
        joint_name = JOINT_MAP.get(angle_name)
        if joint_name and joint_name in self.joint_ids:
            jid = self.joint_ids[joint_name]
            qpos_addr = self.model.jnt_qposadr[jid]
            # Clamp to joint limits
            limited = self.model.jnt_limited[jid]
            if limited:
                lo = self.model.jnt_range[jid][0]
                hi = self.model.jnt_range[jid][1]
                value_rad = np.clip(value_rad, lo, hi)
            self.data.qpos[qpos_addr] = value_rad

    def set_joint_angles(self, angle_dict):
        """Set multiple joints from a {name: radian_value} dict."""
        if angle_dict is None:
            return
        for name, value in angle_dict.items():
            self.set_joint_angle(name, value)

    def set_arm_angles(self, left_angles, right_angles):
        """Convenience: set left and right arm angles from two dicts."""
        if left_angles:
            self.set_joint_angles(left_angles)
        if right_angles:
            self.set_joint_angles(right_angles)

    def set_joint_angle_clipped(self, joint_name, value_deg, max_deg=ANGLE_LIMIT_DEG):
        """Set joint by name with degree-based clamping."""
        rad = np.radians(np.clip(value_deg, -max_deg, max_deg))
        self.set_joint_angle(joint_name, rad)

    def step(self):
        """Advance physics one step."""
        mujoco.mj_step(self.model, self.data)

    def forward(self):
        """Forward kinematics only (no dynamics). Fast, for pure tracking."""
        mujoco.mj_forward(self.model, self.data)

    def render(self, camera_name=None):
        """Render current state to a numpy RGB array."""
        self.renderer.update_scene(self.data, camera=camera_name or "front")
        return self.renderer.render()

    def get_joint_angles(self):
        """Return current joint angles as {name: radian} dict."""
        result = {}
        for name, jid in self.joint_ids.items():
            qpos_addr = self.model.jnt_qposadr[jid]
            result[name] = float(self.data.qpos[qpos_addr])
        return result

    def print_joint_info(self):
        """Print all joint names and their ranges."""
        print("\nG1 Joint Info:")
        print("-" * 60)
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            limited = self.model.jnt_limited[i]
            if limited:
                lo, hi = self.model.jnt_range[i]
                print(f"  {name:35s}  range: [{lo:6.2f}, {hi:6.2f}]")
            else:
                print(f"  {name:35s}  unlimited")
        print("-" * 60)

    def stand_pose(self):
        """Reset robot to neutral standing pose with straight-looking arms."""
        mujoco.mj_resetData(self.model, self.data)
        # Explicitly set all arm joints to neutral
        arm_joints = {
            'shoulder_pitch': 0.0, 'shoulder_roll': 0.0, 'shoulder_yaw': 0.0,
            'elbow': 0.3,  # G1 neutral elbow — looks straight at 0.3 rad, not 0
            'wrist_roll': 0.0, 'wrist_pitch': 0.0, 'wrist_yaw': 0.0,
        }
        for suffix, value in arm_joints.items():
            for side in ['left_', 'right_']:
                jname = side + suffix + '_joint'
                if jname in self.joint_ids:
                    self.data.qpos[self.model.jnt_qposadr[self.joint_ids[jname]]] = value
        self.forward()

    def close(self):
        self.renderer.close()
