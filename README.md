# ObjectFollower v1.1

Real-time color object tracker with proportional controller output.
Press **S**, click on any colored object — it tracks the centroid every frame
and outputs the (dx, dy) pixel error that a pan-tilt servo PID loop consumes directly.

---

## Install

```bash
pip install opencv-python numpy
```

---

## Run

```bash
python object_follower.py
```

Options:

| Flag | Default | What it does |
|---|---|---|
| `--camera N` | `0` | Camera index. Try `1`, `2`... for USB cameras |
| `--kp 0.5` | `0.5` | Proportional gain. Higher = faster response |
| `--min-area 800` | `800` | Ignore blobs smaller than N px2. Increase if tracking noise |
| `--record` | off | Save every frame's error + signal to a timestamped CSV |

---

## Controls

| Key | Action |
|---|---|
| `S` | Enter sampling mode, then click on the target object |
| `R` | Reset / forget current color |
| `+` or `=` | Widen HSV tolerance (tracks more shades of the color) |
| `-` | Narrow HSV tolerance (rejects more background) |
| `Q` or `ESC` | Quit |

---

## What the HUD shows

```
[ TRACKING  hue=15  tol=+-20 ]
Object  : ( 420,  310) px         <- centroid location in the frame
Error   : dx= +100.0  dy=  -50.0  <- offset from frame center
P-signal: pan=+50.00  tilt=-25.00 (clamp +-100)  <- servo command
```

- **Top-left inset** — the HSV mask. Green pixels = what the tracker "sees" as the target.
  Use this to judge whether tolerance is too tight or too wide.
- **Color swatch** (top-right) — the actual color you clicked.
- **Arrow** — drawn from frame center to the detected centroid.

---

## How the P-controller works

```
error_x  =  object_cx  -  frame_center_x     (signed pixels)
error_y  =  object_cy  -  frame_center_y

p_pan    =  Kp * error_x                      (servo command, clamped +-100)
p_tilt   =  Kp * error_y
```

When the object is centered, error = 0 and the signal = 0. Servo holds still.
When the object moves right, p_pan goes positive — the servo pans right to follow.

### Choosing Kp

| Kp | Behavior |
|---|---|
| 0.2-0.3 | Sluggish, won't oscillate |
| 0.4-0.6 | Good starting range |
| 0.8-1.2 | Fast, may overshoot |
| > 1.5 | Likely to oscillate |

Start at 0.5. If the servo hunts back and forth, lower it. If it's too slow, raise it.

---

## Tuning the color range

**Tolerance (+ / - keys)**
- Too narrow: tracking drops out when lighting changes slightly
- Too wide: picks up background of a similar color

If the mask inset shows the wrong region, press R and re-sample from a better-lit part of the object.

If the object is near white or black, the saturation/value floors kick in and reject it.
Try sampling a more saturated area of the object.

---

## Common fixes

| Problem | Fix |
|---|---|
| Tracks wrong color after clicking | Press R then S and click more carefully — avoid edges |
| Centroid jumps to background | Raise --min-area (e.g. --min-area 2000) or narrow tolerance with - |
| Object not found | Increase room brightness or use + to widen tolerance |
| Resolution lower than expected | HUD shows actual resolution. Camera may cap at a lower setting |
| Multiple blobs detected | Script keeps only the largest contour automatically |
| Tracker is slow or choppy | Reduce --min-area or lower RESOLUTION_LADDER in the script |

---

## v2.0 bridge: connecting to a real servo

The p_pan and p_tilt values are exactly what goes into a servo PWM driver:

```python
# Microcontroller side (pseudocode)
servo_pan.duty_cycle  = NEUTRAL + int(p_pan)
servo_tilt.duty_cycle = NEUTRAL - int(p_tilt)   # y-axis often inverted
```

To send over serial, add this inside the tracking block:

```python
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200)
# then inside the loop:
ser.write(f"{p_pan:.1f},{p_tilt:.1f}\n".encode())
```

The entire vision and control logic stays identical. Only the output destination changes.
