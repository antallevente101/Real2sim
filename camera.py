"""Camera capture module."""

import socket
import struct
import cv2
import numpy as np
from g1_tracker.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class Camera:
    """OpenCV camera wrapper."""

    def __init__(self, index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.sock = None
        self.recv_buf = b''

    def open(self):
        # Resolve Windows host IP from WSL2 gateway
        import subprocess, time
        try:
            result = subprocess.run(
                ["sh", "-c", "ip route | grep default | awk '{print $3}'"],
                capture_output=True, text=True, timeout=3
            )
            host_ip = result.stdout.strip() or "172.22.176.1"
        except Exception:
            host_ip = "172.22.176.1"

        # Try connecting; auto-launch Windows camera server if needed
        for attempt in range(2):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5)
                self.sock.connect((host_ip, 9999))
                self.recv_buf = b''
                print(f"Camera connected via TCP bridge ({host_ip}:9999)")
                return self
            except (ConnectionRefusedError, socket.timeout, OSError):
                if self.sock:
                    self.sock.close()
                    self.sock = None
                if attempt == 0:
                    # Auto-start the Windows camera server
                    print("Starting camera server on Windows...")
                    subprocess.Popen(
                        ["cmd.exe", "/c", "start", '""', "python",
                         "\\\\wsl$\\Ubuntu\\home\\levi\\.openclaw\\workspace\\camera_server.py"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    print("Waiting for camera server to start (3s)...")
                    time.sleep(3)
                else:
                    print("ERROR: Camera server did not start. Run manually:")
                    print('  python "\\\\wsl$\\Ubuntu\\home\\levi\\.openclaw\\workspace\\camera_server.py"')
                    raise
        return self  # unreachable

    def read(self):
        """Return (success, frame) or (False, None)."""
        if self.sock is None:
            return False, None
        try:
            # Read 4-byte length prefix
            while len(self.recv_buf) < 4:
                chunk = self.sock.recv(4096)
                if not chunk:
                    return False, None
                self.recv_buf += chunk
            length = struct.unpack('>I', self.recv_buf[:4])[0]
            self.recv_buf = self.recv_buf[4:]
            # Read JPEG payload
            while len(self.recv_buf) < length:
                chunk = self.sock.recv(min(65536, length - len(self.recv_buf)))
                if not chunk:
                    return False, None
                self.recv_buf += chunk
            jpeg_data = self.recv_buf[:length]
            self.recv_buf = self.recv_buf[length:]
            # Decode JPEG to OpenCV frame
            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            return True, frame
        except (socket.timeout, ConnectionError) as e:
            print(f"Camera read error: {e}")
            return False, None

    def release(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.release()
