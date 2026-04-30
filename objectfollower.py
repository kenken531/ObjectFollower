#!/usr/bin/env python3
"""
ObjectFollower v1.1
====================
Color-based object tracker with proportional controller output.
Computes (x, y) pixel offset from frame center — the P-term for a pan-tilt servo loop.

Usage:
    python object_follower.py [--camera 0] [--kp 0.5] [--min-area 800] [--record]

Controls (while running):
    S       — enter sampling mode, then click on the object
    R       — reset / clear target
    + / -   — widen / narrow HSV hue tolerance
    Q / ESC — quit
"""

import argparse
import time
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_CAMERA   = 0
DEFAULT_KP       = 0.5
DEFAULT_MIN_AREA = 800       # px²
DEFAULT_HSV_TOL  = 20        # ±hue tolerance
SAT_MIN          = 70        # reject near-gray pixels
VAL_MIN          = 40        # reject near-black pixels
MAX_SIGNAL       = 100.0

CROSSHAIR_COLOR  = (0, 255, 255)
TARGET_COLOR     = (0, 255, 0)
WARN_COLOR       = (0, 120, 255)
TEXT_COLOR       = (255, 255, 255)

# Resolutions tried in order — camera accepts the first one it supports
RESOLUTION_LADDER = [
    (1920, 1080),
    (1280,  720),
    ( 960,  540),
    ( 640,  480),
]


# ──────────────────────────────────────────────
# Camera setup
# ──────────────────────────────────────────────

def open_camera(index):
    """Open camera at the highest resolution the device supports."""
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {index}.")
        sys.exit(1)

    # Disable camera buffer lag: keep only the freshest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    accepted_w, accepted_h = 0, 0
    for w, h in RESOLUTION_LADDER:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if got_w >= w * 0.9 and got_h >= h * 0.9:
            accepted_w, accepted_h = got_w, got_h
            break
        accepted_w, accepted_h = got_w, got_h

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Camera {index} opened at {accepted_w}x{accepted_h} @ {fps:.0f} fps")
    return cap


# ──────────────────────────────────────────────
# Circular-mean hue  (fixes red-wrap sampling bug)
# ──────────────────────────────────────────────

def circular_mean_hue(hue_channel):
    """
    Correct mean for circular hue (0-179 in OpenCV).
    np.median/mean fail for reds: a patch containing [175,176,1,2,3]
    gives median 90 (green) instead of ~0/179 (red).
    We project hues onto the unit circle, average, then project back.
    """
    h = hue_channel.flatten().astype(np.float32)
    angles = h * (2.0 * np.pi / 180.0)
    sin_m  = np.mean(np.sin(angles))
    cos_m  = np.mean(np.cos(angles))
    mean_a = np.arctan2(sin_m, cos_m)
    if mean_a < 0:
        mean_a += 2.0 * np.pi
    return int(round(mean_a * 180.0 / (2.0 * np.pi))) % 180


# ──────────────────────────────────────────────
# HSV masking
# ──────────────────────────────────────────────

def build_mask(hsv_frame, hue, tol, sat_min, val_min):
    """
    Binary mask for target hue. Handles the 0/179 red wrap-around automatically.
    """
    lo_h = (hue - tol) % 180
    hi_h = (hue + tol) % 180

    if lo_h <= hi_h:
        mask = cv2.inRange(
            hsv_frame,
            np.array([lo_h, sat_min, val_min]),
            np.array([hi_h, 255,     255    ]),
        )
    else:
        # Wrap-around: two ranges OR-ed together
        mask = (
            cv2.inRange(hsv_frame,
                        np.array([lo_h, sat_min, val_min]),
                        np.array([179,  255,     255    ])) |
            cv2.inRange(hsv_frame,
                        np.array([0,    sat_min, val_min]),
                        np.array([hi_h, 255,     255    ]))
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ──────────────────────────────────────────────
# Contour + centroid
# ──────────────────────────────────────────────

def largest_contour(mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    return best if cv2.contourArea(best) >= min_area else None


def centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


# ──────────────────────────────────────────────
# Overlay drawing
# ──────────────────────────────────────────────

def draw_crosshair(frame, cx, cy, size=20, color=CROSSHAIR_COLOR, thickness=2):
    cv2.line(  frame, (cx - size, cy),      (cx + size, cy),      color, thickness)
    cv2.line(  frame, (cx,        cy-size), (cx,        cy+size), color, thickness)
    cv2.circle(frame, (cx, cy), size // 2,  color, thickness)


def draw_hud(frame, state):
    fh, fw = frame.shape[:2]
    fcx, fcy = fw // 2, fh // 2

    draw_crosshair(frame, fcx, fcy, size=14, color=CROSSHAIR_COLOR, thickness=1)

    bar_h = 115
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, fh - bar_h), (fw, fh), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    if state["tracking"]:
        ex, ey = state["error"]
        px, py = state["p_signal"]
        ox, oy = state["obj_center"]

        cv2.putText(frame,
                    f"[ TRACKING  hue={state['hue']}  tol=\xb1{state['tol']} ]",
                    (14, fh - 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, TARGET_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"Object  : ({ox:4d}, {oy:4d}) px",
                    (14, fh - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"Error   : dx={ex:+6.1f}  dy={ey:+6.1f} px",
                    (14, fh - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"P-signal: pan={px:+6.2f}  tilt={py:+6.2f}  (clamp \xb1{MAX_SIGNAL:.0f})",
                    (14, fh - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 220, 255), 1, cv2.LINE_AA)

        cv2.arrowedLine(frame, (fcx, fcy), (ox, oy), (50, 200, 50), 2, tipLength=0.12)

    else:
        hue_str = f"hue={state['hue']}  tol=\xb1{state['tol']}" \
                  if state["hue"] is not None else "no target"
        cv2.putText(frame,
                    f"[ NO TARGET  {hue_str} ]",
                    (14, fh - 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, WARN_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"Min area : {state['min_area']} px\xb2",
                    (14, fh - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    "S = sample color   R = reset   +/- = tolerance   Q = quit",
                    (14, fh - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    f"Resolution: {fw}x{fh}",
                    (14, fh - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.putText(frame, f"{state['fps']:.0f} fps",
                (fw - 80, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (70, 70, 70), 1, cv2.LINE_AA)

    # Color swatch showing the actual sampled color
    if state["hue"] is not None:
        sc = state.get("sampled_bgr")
        if sc:
            swatch_color = (int(sc[0]), int(sc[1]), int(sc[2]))
        else:
            bgr = cv2.cvtColor(np.uint8([[[state["hue"], 200, 200]]]),
                                cv2.COLOR_HSV2BGR)[0][0]
            swatch_color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        swatch = np.zeros((22, 56, 3), dtype=np.uint8)
        swatch[:] = swatch_color
        frame[30:52, fw - 64: fw - 8] = swatch
        cv2.rectangle(frame, (fw - 64, 30), (fw - 8, 52), (180, 180, 180), 1)


# ──────────────────────────────────────────────
# Mouse callback
# ──────────────────────────────────────────────

_click = {"pending": False, "x": 0, "y": 0}

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and param["sampling"]:
        _click["pending"] = True
        _click["x"] = x
        _click["y"] = y


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def run(camera_idx, kp, min_area, record):
    cap = open_camera(camera_idx)

    win_name = "ObjectFollower v1.1"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    state = {
        "hue":         None,
        "sampled_bgr": None,
        "tol":         DEFAULT_HSV_TOL,
        "sat_min":     SAT_MIN,
        "val_min":     VAL_MIN,
        "min_area":    min_area,
        "sampling":    False,
        "tracking":    False,
        "error":       (0.0, 0.0),
        "p_signal":    (0.0, 0.0),
        "obj_center":  (0, 0),
        "fps":         0.0,
    }

    cv2.setMouseCallback(win_name, on_mouse, state)

    log_file = log_writer = None
    if record:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path  = Path(f"tracking_log_{ts}.csv")
        log_file  = open(log_path, "w", newline="")
        log_writer= csv.writer(log_file)
        log_writer.writerow(["timestamp", "obj_x", "obj_y",
                              "error_x", "error_y", "p_pan", "p_tilt"])
        print(f"[INFO] Logging to {log_path}")

    print("[INFO] Press S, then click on the colored object to start tracking.")
    print("       R=reset  +/-=tolerance  Q/ESC=quit")

    prev_t = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — retrying...")
            time.sleep(0.03)
            continue

        fh, fw = frame.shape[:2]
        fcx, fcy = fw // 2, fh // 2

        # FPS
        now    = time.perf_counter()
        dt     = now - prev_t
        state["fps"] = 1.0 / dt if dt > 0 else 0.0
        prev_t = now

        # ── Color sample on click ─────────────────────────────────────────────
        if _click["pending"] and state["sampling"]:
            _click["pending"] = False
            px, py = _click["x"], _click["y"]

            r  = 5   # sample an 11x11 patch
            y0, y1 = max(0, py - r), min(fh, py + r + 1)
            x0, x1 = max(0, px - r), min(fw, px + r + 1)
            patch_bgr = frame[y0:y1, x0:x1]

            if patch_bgr.size > 0:
                patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)

                # Circular mean — the only correct way to average hues
                hue   = circular_mean_hue(patch_hsv[:, :, 0])
                med_s = int(np.median(patch_hsv[:, :, 1]))
                med_v = int(np.median(patch_hsv[:, :, 2]))

                # Store actual clicked pixel BGR for the swatch
                cy_idx = patch_bgr.shape[0] // 2
                cx_idx = patch_bgr.shape[1] // 2
                state["sampled_bgr"] = patch_bgr[cy_idx, cx_idx].tolist()

                # Adaptive sat/val floors derived from what was actually sampled
                state["sat_min"] = max(SAT_MIN, int(med_s * 0.60))
                state["val_min"] = max(VAL_MIN, int(med_v * 0.60))
                state["hue"]     = hue
                state["sampling"]= False

                print(f"[INFO] Sampled hue={hue}  sat_floor={state['sat_min']}"
                      f"  val_floor={state['val_min']}  at ({px},{py})")

        # ── Tracking ──────────────────────────────────────────────────────────
        state["tracking"] = False

        if state["hue"] is not None:
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = build_mask(hsv, state["hue"], state["tol"],
                              state["sat_min"], state["val_min"])
            cnt  = largest_contour(mask, state["min_area"])

            if cnt is not None:
                cx_obj, cy_obj = centroid(cnt)
                if cx_obj is not None:
                    ex = float(cx_obj - fcx)
                    ey = float(cy_obj - fcy)
                    p_pan  = max(-MAX_SIGNAL, min(MAX_SIGNAL, kp * ex))
                    p_tilt = max(-MAX_SIGNAL, min(MAX_SIGNAL, kp * ey))

                    state["tracking"]   = True
                    state["error"]      = (ex, ey)
                    state["p_signal"]   = (p_pan, p_tilt)
                    state["obj_center"] = (cx_obj, cy_obj)

                    cv2.drawContours(frame, [cnt], -1, TARGET_COLOR, 2)
                    draw_crosshair(frame, cx_obj, cy_obj, size=20, color=TARGET_COLOR)

                    if log_writer:
                        log_writer.writerow([
                            f"{time.time():.4f}",
                            cx_obj, cy_obj,
                            f"{ex:.2f}", f"{ey:.2f}",
                            f"{p_pan:.4f}", f"{p_tilt:.4f}",
                        ])

            # HSV mask inset (top-left)
            iw = min(200, fw // 4)
            ih = min(150, fh // 4)
            mask_small = cv2.resize(mask, (iw, ih))
            mask_bgr   = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            mask_bgr[mask_small > 0] = [0, 200, 80]   # tint detected area green
            frame[0:ih, 0:iw] = (mask_bgr.astype(np.uint16) * 6 // 10).astype(np.uint8)
            cv2.rectangle(frame, (0, 0), (iw, ih), (60, 60, 60), 1)
            cv2.putText(frame, "HSV mask", (4, ih + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 80, 80), 1, cv2.LINE_AA)

        # ── Sampling mode border ──────────────────────────────────────────────
        if state["sampling"]:
            cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (0, 160, 255), 4)
            msg = "CLICK on the object to sample its color"
            tw  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
            cv2.putText(frame, msg,
                        ((fw - tw) // 2, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 160, 255), 2, cv2.LINE_AA)

        draw_hud(frame, state)
        cv2.imshow(win_name, frame)

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('s'), ord('S')):
            state["sampling"] = True
            print("[INFO] Sampling mode ON — click on the object.")
        elif key in (ord('r'), ord('R')):
            state.update(hue=None, tracking=False, sampling=False,
                         sampled_bgr=None, sat_min=SAT_MIN, val_min=VAL_MIN)
            print("[INFO] Reset.")
        elif key in (ord('+'), ord('=')):
            state["tol"] = min(89, state["tol"] + 5)
            print(f"[INFO] Tolerance: +/-{state['tol']}")
        elif key == ord('-'):
            state["tol"] = max(5, state["tol"] - 5)
            print(f"[INFO] Tolerance: +/-{state['tol']}")

    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
        print("[INFO] Log saved.")
    print("[INFO] Done.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ObjectFollower v1.1")
    ap.add_argument("--camera",   type=int,   default=DEFAULT_CAMERA)
    ap.add_argument("--kp",       type=float, default=DEFAULT_KP,
                    help="Proportional gain (default 0.5)")
    ap.add_argument("--min-area", type=int,   default=DEFAULT_MIN_AREA,
                    help="Min contour area px^2 (default 800)")
    ap.add_argument("--record",   action="store_true",
                    help="Log tracking data to CSV")
    args = ap.parse_args()

    run(args.camera, args.kp, args.min_area, args.record)