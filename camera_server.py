r"""
Windows-side camera server — captures frames and sends them to WSL via TCP.

Prerequisites on Windows:
  pip install opencv-python

⚠️  Windows Defender Firewall will prompt on first run — click ALLOW.
   If you miss it: Windows Settings → Firewall → Allow an app → Python

Usage:
  python \\wsl$\Ubuntu\home\levi\.openclaw\workspace\camera_server.py
"""
import socket
import struct
import cv2

HOST = '127.0.0.1'
PORT = 9999
WIDTH = 640
HEIGHT = 480
FPS = 30

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap.isOpened():
    print("ERROR: Cannot open camera. Is another app using it?")
    exit(1)

print(f"Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', PORT))  # 0.0.0.0 = reachable from WSL2
server.listen(1)
print(f"Waiting for WSL client on {HOST}:{PORT}...")

conn, addr = server.accept()
conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # disable Nagle, send immediately
print(f"Client connected from {addr}")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            break
        # Flip handled in WSL (MIRROR_CAMERA) — don't double-flip
        # Encode as JPEG — quality 50 is fast + small enough for MediaPipe
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        # Send: 4-byte length prefix + JPEG data
        conn.sendall(struct.pack('>I', len(jpeg)) + jpeg.tobytes())
except (BrokenPipeError, ConnectionResetError):
    print("Client disconnected")
finally:
    conn.close()
    server.close()
    cap.release()
    print("Done")
