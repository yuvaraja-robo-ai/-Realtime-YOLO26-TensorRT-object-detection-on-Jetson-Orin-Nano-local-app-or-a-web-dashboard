#!/usr/bin/env python3
"""Verify the USB camera opens and reports a real framerate.

Run before building on the camera:
    python3 scripts/check_camera.py            # default /dev/video0 @ 1280x720
    python3 scripts/check_camera.py 0 1920 1080
"""
import sys
import time

import cv2


def main() -> int:
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 720

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"FAIL: could not open /dev/video{dev}")
        return 1

    # Force MJPG — required to hit 30fps at HD on most USB UVC cams.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"opened /dev/video{dev}  requested {width}x{height}  actual {aw}x{ah}")

    n, ok_frames, t0 = 60, 0, time.time()
    for _ in range(n):
        ok, frame = cap.read()
        if ok and frame is not None:
            ok_frames += 1
    dt = time.time() - t0
    cap.release()

    if ok_frames == 0:
        print("FAIL: zero frames captured")
        return 1
    fps = ok_frames / dt if dt > 0 else 0
    print(f"captured {ok_frames}/{n} frames in {dt:.2f}s  ->  {fps:.1f} FPS")
    print("OK" if fps > 5 else "WARN: low FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
