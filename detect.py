import cv2
import time
import threading
from pathlib import Path
from ultralytics import YOLO

from detect_and_upload import handle_detection, handle_disease_classification

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent

# Fix #1: MODEL_FIRE now auto-discovers the latest phase2 run from train_fire.py output,
# falling back to models/fire_best.pt if no training run exists yet.
def _find_fire_model():
    runs_dir = SCRIPT_DIR / "training_runs"
    if runs_dir.exists():
        candidates = sorted(runs_dir.glob("*_phase2/weights/best.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            print(f"[fire] Auto-detected model: {candidates[0]}")
            return candidates[0]
    fallback = SCRIPT_DIR / "models" / "fire_best.pt"
    return fallback

MODEL_FIRE  = _find_fire_model()
MODEL_PLANT = SCRIPT_DIR / "models" / "plant_best.pt"

# ── Upload thresholds ──────────────────────────────────────────────
CONF_UPLOAD   = 0.80
CONFIRM_SECS  = 2.0
SAVE_COOLDOWN = 30.0

# ── Fire model (object detection, 640px) ──────────────────────────
FIRE_CONF_DISPLAY  = 0.35
FIRE_IMGSZ         = 640
FIRE_IGNORE        = {"default"}
FIRE_MAX_AREA      = 0.65   # reject boxes covering > 65% of frame (background walls)
FIRE_LEFT_MARGIN   = 8      # reject boxes starting at left edge (wall bleed-in)
FIRE_STICKY_FRAMES = 6      # keep showing last box for N frames after detection lost

# ── Plant model (classification, 256px) ───────────────────────────
PLANT_CONF_DISPLAY = 0.60
PLANT_IMGSZ        = 256
PLANT_GREEN_THRESH = 0.08

# ── GPS — Fix #2: replace hardcoded placeholders with configurable values ──
# Set these to your actual location or hook up a real GPS module.
GPS_LAT = None   # e.g. 32.0853 — set before running
GPS_LNG = None   # e.g. 34.7818

def _gps():
    if GPS_LAT is None or GPS_LNG is None:
        print("[WARNING] GPS_LAT / GPS_LNG not set — uploads will use 0.0, 0.0")
        return 0.0, 0.0
    return GPS_LAT, GPS_LNG

# ── Shared: raw camera frame ──────────────────────────────────────
_raw_frame    = None
_raw_frame_id = 0
_raw_lock     = threading.Lock()

# Fix #5: camera-ready event so workers don't busy-loop before first frame
_camera_ready = threading.Event()

# ── Shared: annotation data (workers write, main draws) ───────────
_fire_boxes  = []      # list of [x1, y1, x2, y2, label, conf]
_fire_alock  = threading.Lock()

# [0] = (label, conf, veg_bbox) or None
_plant_state = [None]
_plant_alock = threading.Lock()

_stop = threading.Event()


# ─────────────────────────────────────────────────────────────────
# FIREBASE UPLOAD
# ─────────────────────────────────────────────────────────────────
def upload_to_firebase(model_name, label, confidence, frame, bbox):
    lat, lng = _gps()
    try:
        if model_name == "fire":
            handle_detection(
                frame=frame,
                bbox=bbox,
                anomaly_type="fire",
                confidence=confidence,
                gps_lat=lat,
                gps_lng=lng,
                label=label,
            )
            print(f"[Firebase] fire uploaded: {label} {confidence:.0%}")
        elif model_name == "plant":
            handle_disease_classification(
                frame=frame,
                disease_name=label,
                confidence=confidence,
                gps_lat=lat,
                gps_lng=lng,
                bbox=None,
            )
            print(f"[Firebase] plant uploaded: {label} {confidence:.0%}")
    except Exception as exc:
        print(f"[Firebase ERROR] {model_name}: {exc}")


# ─────────────────────────────────────────────────────────────────
# FIRE WORKER
# ─────────────────────────────────────────────────────────────────
def _fire_worker():
    print("[fire] Loading model...")
    model = YOLO(str(MODEL_FIRE))
    print(f"[fire] Ready. Classes: {list(model.names.values())}")

    # Fix #5: wait for camera before entering the detection loop
    _camera_ready.wait()

    last_seen_id    = -1
    first_detected  = None
    first_label     = None   # Fix #3: track which label started the timer
    last_uploaded   = {}     # Fix #4: per-label cooldown dict
    missed_frames   = 0
    last_best_box   = None

    while not _stop.is_set():
        with _raw_lock:
            fid   = _raw_frame_id
            frame = _raw_frame.copy() if _raw_frame is not None else None

        if frame is None or fid == last_seen_id:
            time.sleep(0.005)
            continue
        last_seen_id = fid

        fh, fw     = frame.shape[:2]
        frame_area = fw * fh

        results = model(frame, conf=FIRE_CONF_DISPLAY, imgsz=FIRE_IMGSZ, verbose=False)
        result  = results[0]

        valid = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label  = result.names[cls_id]
            if label.lower() in FIRE_IGNORE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if (x2 - x1) * (y2 - y1) / frame_area > FIRE_MAX_AREA:
                continue
            if x1 <= FIRE_LEFT_MARGIN:
                continue
            valid.append([x1, y1, x2, y2, label, conf])

        if valid:
            best_box      = max(valid, key=lambda b: b[5])
            last_best_box = best_box
            missed_frames = 0
            boxes_data    = [best_box]
        else:
            missed_frames += 1
            if missed_frames <= FIRE_STICKY_FRAMES and last_best_box is not None:
                boxes_data = [last_best_box]
            else:
                boxes_data    = []
                last_best_box = None

        with _fire_alock:
            _fire_boxes.clear()
            _fire_boxes.extend(boxes_data)

        now = time.time()

        if not valid and missed_frames > FIRE_STICKY_FRAMES:
            if first_detected is not None:
                print("[fire] Lost — timer reset")
            first_detected = None
            first_label    = None
            continue

        if not valid:
            continue

        best = max(valid, key=lambda b: b[5])
        x1, y1, x2, y2, best_label, best_conf = best
        bbox = [x1, y1, x2, y2]

        # Fix #3: reset timer if the dominant label changed mid-confirmation
        if first_label is not None and best_label != first_label:
            print(f"[fire] Label changed {first_label} → {best_label} — timer reset")
            first_detected = None
            first_label    = None

        if first_detected is None:
            first_detected = now
            first_label    = best_label
            print(f"[fire] Detected {best_label} {best_conf:.0%} — confirming...")

        if best_conf < CONF_UPLOAD:
            elapsed = now - first_detected
            print(f"[fire] Confirming... {elapsed:.1f}s / {CONFIRM_SECS}s (conf {best_conf:.0%})")
            continue

        elapsed          = now - first_detected
        label_last_saved = last_uploaded.get(best_label, 0.0)  # Fix #4: per-label cooldown

        if elapsed >= CONFIRM_SECS and (now - label_last_saved) >= SAVE_COOLDOWN:
            print(f"[fire] Confirmed {elapsed:.1f}s — uploading...")
            threading.Thread(
                target=upload_to_firebase,
                args=("fire", best_label, best_conf, frame, bbox),
                daemon=True,
            ).start()
            last_uploaded[best_label] = now
            first_detected = None
            first_label    = None
        elif elapsed < CONFIRM_SECS:
            print(f"[fire] Confirming... {elapsed:.1f}s / {CONFIRM_SECS}s")
        else:
            remaining = SAVE_COOLDOWN - (now - label_last_saved)
            print(f"[fire] Confirmed but cooling down — {remaining:.0f}s remaining")


# ─────────────────────────────────────────────────────────────────
# PLANT WORKER
# ─────────────────────────────────────────────────────────────────
def _plant_worker():
    print("[plant] Loading model...")
    model = YOLO(str(MODEL_PLANT))
    print(f"[plant] Ready. Classes: {list(model.names.values())}")

    # Fix #5: wait for camera before entering the detection loop
    _camera_ready.wait()

    last_seen_id   = -1
    first_detected = None
    first_label    = None   # Fix #3
    last_uploaded  = {}     # Fix #4: per-label cooldown dict

    while not _stop.is_set():
        with _raw_lock:
            fid   = _raw_frame_id
            frame = _raw_frame.copy() if _raw_frame is not None else None

        if frame is None or fid == last_seen_id:
            time.sleep(0.005)
            continue
        last_seen_id = fid

        hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask  = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = float(green_mask.sum()) / 255.0 / (frame.shape[0] * frame.shape[1])

        if green_ratio < PLANT_GREEN_THRESH:
            with _plant_alock:
                _plant_state[0] = None
            if first_detected is not None:
                print("[plant] Not green enough — timer reset")
            first_detected = None
            first_label    = None
            continue

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        veg_bbox = None
        if contours:
            largest        = max(contours, key=cv2.contourArea)
            gx, gy, gw, gh = cv2.boundingRect(largest)
            veg_bbox       = (gx, gy, gx + gw, gy + gh)

        results = model(frame, imgsz=PLANT_IMGSZ, verbose=False)
        result  = results[0]

        if result.probs is None:
            continue

        top1_idx  = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)
        label     = result.names[top1_idx]

        now = time.time()

        if top1_conf < PLANT_CONF_DISPLAY:
            with _plant_alock:
                _plant_state[0] = None
            if first_detected is not None:
                print("[plant] Low conf — timer reset")
            first_detected = None
            first_label    = None
            continue

        with _plant_alock:
            _plant_state[0] = (label, top1_conf, veg_bbox)

        # Fix #3: reset timer if label changed mid-confirmation
        if first_label is not None and label != first_label:
            print(f"[plant] Label changed {first_label} → {label} — timer reset")
            first_detected = None
            first_label    = None

        if first_detected is None:
            first_detected = now
            first_label    = label
            print(f"[plant] Detected {label} {top1_conf:.0%} — confirming...")

        if top1_conf < CONF_UPLOAD:
            elapsed = now - first_detected
            print(f"[plant] Confirming... {elapsed:.1f}s / {CONFIRM_SECS}s (conf {top1_conf:.0%})")
            continue

        elapsed          = now - first_detected
        label_last_saved = last_uploaded.get(label, 0.0)  # Fix #4: per-label cooldown

        if elapsed >= CONFIRM_SECS and (now - label_last_saved) >= SAVE_COOLDOWN:
            print(f"[plant] Confirmed {elapsed:.1f}s — uploading...")
            threading.Thread(
                target=upload_to_firebase,
                args=("plant", label, top1_conf, frame, None),
                daemon=True,
            ).start()
            last_uploaded[label] = now
            first_detected       = None
            first_label          = None
        elif elapsed < CONFIRM_SECS:
            print(f"[plant] Confirming... {elapsed:.1f}s / {CONFIRM_SECS}s")
        else:
            remaining = SAVE_COOLDOWN - (now - label_last_saved)
            print(f"[plant] Confirmed but cooling down — {remaining:.0f}s remaining")


# ─────────────────────────────────────────────────────────────────
# DRAW  — called by main on a fresh frame every cycle
# ─────────────────────────────────────────────────────────────────
def _draw(frame):
    with _fire_alock:
        boxes = list(_fire_boxes)

    for x1, y1, x2, y2, label, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 60, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)

    with _plant_alock:
        plant = _plant_state[0]

    if plant is not None:
        label, conf, veg_bbox = plant

        if veg_bbox is not None:
            px1, py1, px2, py2 = veg_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 60), 2)

        clean = label.replace("___", " ").replace("_", " ")
        text  = f"{clean}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (8, 8), (8 + tw + 10, 8 + th + 12), (0, 0, 0), -1)
        cv2.putText(frame, text, (13, 8 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 60), 2)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    global _raw_frame, _raw_frame_id

    for name, path in [("Fire", MODEL_FIRE), ("Plant", MODEL_PLANT)]:
        if not path.exists():
            print(f"[ERROR] {name} model not found: {path}")
            return

    if GPS_LAT is None or GPS_LNG is None:
        print("[WARNING] GPS coordinates not set — edit GPS_LAT / GPS_LNG in detect.py")

    threading.Thread(target=_fire_worker,  daemon=True).start()
    threading.Thread(target=_plant_worker, daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        _stop.set()
        _camera_ready.set()   # unblock workers so they see _stop and exit cleanly
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Fix #5: signal workers only after camera is confirmed open
    _camera_ready.set()

    print("\nAgroSentinel running — press Q to quit.")
    print(f"Fire model       : {MODEL_FIRE}")
    print(f"Upload threshold : {CONF_UPLOAD:.0%}")
    print(f"Confirm window   : {CONFIRM_SECS}s")
    print(f"Save cooldown    : {SAVE_COOLDOWN}s\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        with _raw_lock:
            _raw_frame     = frame.copy()
            _raw_frame_id += 1

        _draw(frame)
        cv2.imshow("AgroSentinel", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    _stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
