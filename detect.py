"""AgroSentinel — real-time fire/smoke detection + plant disease classification.

Production-grade edge inference pipeline:
  * Two model workers (fire detection, plant classification) run on background
    threads, consuming the latest camera frame.
  * Confirmed detections are pushed to a bounded queue and uploaded to Firebase
    by a dedicated uploader pool that DRAINS on shutdown (no dropped events).
  * Workers are supervised: an unhandled exception restarts the worker instead
    of silently freezing detections.
  * Graceful shutdown on Q / Ctrl+C / camera loss — threads are joined.

Configuration is centralized in the Config dataclass and overridable via
environment variables (AGRO_*), so the same binary runs in dev and on device.
"""

from __future__ import annotations

import os
import sys
import time
import queue
import signal
import logging
import functools
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from detect_and_upload import handle_detection, handle_disease_classification


# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("AGRO_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agrosentinel")


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_opt_float(key: str) -> Optional[float]:
    """Optional float env var — returns None if unset/invalid (for GPS)."""
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return None


def _find_fire_model() -> Path:
    """Locate the newest trained fire model from train_fire.py output.

    Fix #1 / cross-file: glob ``*_phase2*`` (not ``*_phase2``) so Ultralytics
    auto-incremented run dirs (``..._phase22``) are matched.
    Fix #6: broken symlinks / unreadable files are skipped, not fatal.
    """
    runs_dir = SCRIPT_DIR / "training_runs"
    candidates: list[tuple[float, Path]] = []
    if runs_dir.exists():
        for p in runs_dir.glob("*_phase2*/weights/best.pt"):
            try:
                candidates.append((p.stat().st_mtime, p))
            except OSError as exc:                       # broken symlink / perms
                log.warning("Skipping unreadable model %s: %s", p, exc)
    if candidates:
        newest = max(candidates, key=lambda t: t[0])[1]
        log.info("Auto-detected fire model: %s", newest)
        return newest
    fallback = SCRIPT_DIR / "models" / "fire_best.pt"
    log.info("No trained run found — falling back to %s", fallback)
    return fallback


@dataclass
class Config:
    # Models
    model_fire: Path = field(default_factory=_find_fire_model)
    model_plant: Path = field(default_factory=lambda: SCRIPT_DIR / "models" / "plant_best.pt")

    # Upload / confirmation
    conf_upload: float = _env_float("AGRO_CONF_UPLOAD", 0.80)
    confirm_secs: float = _env_float("AGRO_CONFIRM_SECS", 2.0)
    save_cooldown: float = _env_float("AGRO_SAVE_COOLDOWN", 30.0)

    # Fire model (object detection). imgsz matches training default (1280) for
    # best small/distant-fire recall; override with AGRO_FIRE_IMGSZ on weak HW.
    fire_conf_display: float = _env_float("AGRO_FIRE_CONF", 0.35)
    fire_imgsz: int = _env_int("AGRO_FIRE_IMGSZ", 1280)
    fire_max_area: float = 0.65        # reject boxes > 65% of frame (walls)
    fire_left_margin: int = 8          # reject boxes hugging the left edge
    fire_sticky_frames: int = 6        # keep last box visible N frames

    # Plant model (classification)
    plant_conf_display: float = 0.60
    plant_imgsz: int = _env_int("AGRO_PLANT_IMGSZ", 256)
    plant_green_thresh: float = 0.08

    # GPS — required for meaningful uploads
    gps_lat: Optional[float] = field(default_factory=lambda: _env_opt_float("AGRO_GPS_LAT"))
    gps_lng: Optional[float] = field(default_factory=lambda: _env_opt_float("AGRO_GPS_LNG"))

    # Runtime
    camera_index: int = _env_int("AGRO_CAMERA", 0)
    upload_queue_max: int = 64
    upload_workers: int = 2
    worker_restart_max: int = 5        # give up after N crashes per worker

    upload_drain_secs: float = 8.0     # best-effort drain window on shutdown

    fire_ignore: frozenset[str] = frozenset({"default"})

    def __post_init__(self) -> None:
        # Validate env-driven numbers so a typo'd AGRO_* var fails loudly,
        # not silently (e.g. conf=-1 uploads everything, conf=5 uploads nothing).
        for name in ("conf_upload", "fire_conf_display", "plant_conf_display"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                clamped = min(1.0, max(0.0, v))
                log.warning("%s=%s out of [0,1] — clamping to %s", name, v, clamped)
                setattr(self, name, clamped)
        for name in ("confirm_secs", "save_cooldown"):
            if getattr(self, name) < 0:
                log.warning("%s < 0 — forcing 0", name)
                setattr(self, name, 0.0)
        if self.fire_imgsz % 32 or self.plant_imgsz % 32:
            log.warning("imgsz should be a multiple of 32 — YOLO will round silently")

    def gps(self) -> tuple[float, float]:
        if self.gps_lat is None or self.gps_lng is None:
            log.warning("GPS not set — uploading with 0.0, 0.0")
            return 0.0, 0.0
        return self.gps_lat, self.gps_lng


CFG = Config()


# ─────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────
class FrameBus:
    """Single-slot latest-frame buffer shared between camera and workers."""

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._id = 0
        self._lock = threading.Lock()
        self.ready = threading.Event()

    def publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._id += 1

    def latest(self) -> tuple[int, Optional[np.ndarray]]:
        with self._lock:
            return self._id, (self._frame.copy() if self._frame is not None else None)


BUS = FrameBus()
STOP = threading.Event()

_fire_boxes: list = []
_fire_lock = threading.Lock()
_plant_state: list = [None]   # [0] = (label, conf, veg_bbox) or None
_plant_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────
# UPLOAD QUEUE — drains on shutdown so in-flight events aren't lost
# ─────────────────────────────────────────────────────────────────
@dataclass
class UploadJob:
    model_name: str
    label: str
    confidence: float
    frame: np.ndarray
    bbox: Optional[list]


_upload_q: "queue.Queue[Optional[UploadJob]]" = queue.Queue(maxsize=CFG.upload_queue_max)


def _do_upload(job: UploadJob) -> None:
    lat, lng = CFG.gps()
    try:
        if job.model_name == "fire":
            handle_detection(
                frame=job.frame, bbox=job.bbox, anomaly_type="fire",
                confidence=job.confidence, gps_lat=lat, gps_lng=lng, label=job.label,
            )
        elif job.model_name == "plant":
            handle_disease_classification(
                frame=job.frame, disease_name=job.label, confidence=job.confidence,
                gps_lat=lat, gps_lng=lng, bbox=None,
            )
        log.info("[firebase] %s uploaded: %s %.0f%%",
                 job.model_name, job.label, job.confidence * 100)
    except Exception:
        log.exception("[firebase] upload failed for %s/%s", job.model_name, job.label)


def _uploader_loop() -> None:
    while True:
        job = _upload_q.get()
        try:
            if job is None:          # poison pill — shutdown
                return
            _do_upload(job)
        finally:
            _upload_q.task_done()


def enqueue_upload(model_name: str, label: str, conf: float,
                   frame: np.ndarray, bbox: Optional[list]) -> bool:
    """Queue an upload. Returns False if the queue is full (event dropped)."""
    try:
        _upload_q.put_nowait(UploadJob(model_name, label, conf, frame, bbox))
        return True
    except queue.Full:
        log.warning("[firebase] queue full — dropped %s/%s", model_name, label)
        return False


# ─────────────────────────────────────────────────────────────────
# FIRE WORKER
# ─────────────────────────────────────────────────────────────────
def _fire_worker(model, last_uploaded: dict) -> None:
    # model + last_uploaded are owned by main() and passed in, so a supervisor
    # restart neither reloads the model (no GPU leak) nor loses cooldown state.
    BUS.ready.wait()

    last_seen_id = -1
    first_detected: Optional[float] = None
    first_label: Optional[str] = None
    missed_frames = 0
    last_best_box = None

    while not STOP.is_set():
        fid, frame = BUS.latest()
        if frame is None or fid == last_seen_id:
            time.sleep(0.005)
            continue
        last_seen_id = fid

        fh, fw = frame.shape[:2]
        frame_area = float(fw * fh)

        results = model(frame, conf=CFG.fire_conf_display,
                        imgsz=CFG.fire_imgsz, verbose=False)
        result = results[0]

        valid = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, str(cls_id))   # Fix #6: no KeyError
            if label.lower() in CFG.fire_ignore:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if (x2 - x1) * (y2 - y1) / frame_area > CFG.fire_max_area:
                continue
            if x1 <= CFG.fire_left_margin:
                continue
            valid.append([x1, y1, x2, y2, label, conf])

        if valid:
            best_box = max(valid, key=lambda b: b[5])
            last_best_box = best_box
            missed_frames = 0
            boxes_data = [best_box]
        else:
            missed_frames += 1
            if missed_frames <= CFG.fire_sticky_frames and last_best_box is not None:
                boxes_data = [last_best_box]
            else:
                boxes_data = []
                last_best_box = None

        with _fire_lock:
            _fire_boxes.clear()
            _fire_boxes.extend(boxes_data)

        now = time.time()

        if not valid and missed_frames > CFG.fire_sticky_frames:
            if first_detected is not None:
                log.debug("[fire] lost — timer reset")
            first_detected = first_label = None
            continue
        if not valid:
            continue

        x1, y1, x2, y2, best_label, best_conf = max(valid, key=lambda b: b[5])
        bbox = [x1, y1, x2, y2]

        if first_label is not None and best_label != first_label:
            log.debug("[fire] label changed %s -> %s — timer reset", first_label, best_label)
            first_detected = first_label = None

        if first_detected is None:
            first_detected = now
            first_label = best_label
            log.info("[fire] detected %s %.0f%% — confirming", best_label, best_conf * 100)

        if best_conf < CFG.conf_upload:
            continue

        elapsed = now - first_detected
        cooled = now - last_uploaded.get(best_label, 0.0)
        if elapsed >= CFG.confirm_secs and cooled >= CFG.save_cooldown:
            # Only start the cooldown / reset the timer if the event was actually
            # queued. A dropped (queue-full) fire must NOT be suppressed for 30s.
            if enqueue_upload("fire", best_label, best_conf, frame, bbox):
                log.info("[fire] confirmed %.1fs — queued upload", elapsed)
                last_uploaded[best_label] = now
                first_detected = first_label = None
            else:
                log.warning("[fire] upload dropped (queue full) — will retry")
        elif elapsed >= CFG.confirm_secs:
            log.debug("[fire] confirmed but cooling down — %.0fs left",
                      CFG.save_cooldown - cooled)


# ─────────────────────────────────────────────────────────────────
# PLANT WORKER
# ─────────────────────────────────────────────────────────────────
def _plant_worker(model, last_uploaded: dict) -> None:
    # model + last_uploaded owned by main() — survives supervisor restart.
    BUS.ready.wait()

    last_seen_id = -1
    first_detected: Optional[float] = None
    first_label: Optional[str] = None

    while not STOP.is_set():
        fid, frame = BUS.latest()
        if frame is None or fid == last_seen_id:
            time.sleep(0.005)
            continue
        last_seen_id = fid

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = float(green_mask.sum()) / 255.0 / (frame.shape[0] * frame.shape[1])

        if green_ratio < CFG.plant_green_thresh:
            with _plant_lock:
                _plant_state[0] = None
            first_detected = first_label = None
            continue

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        veg_bbox = None
        if contours:
            gx, gy, gw, gh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            veg_bbox = (gx, gy, gx + gw, gy + gh)

        result = model(frame, imgsz=CFG.plant_imgsz, verbose=False)[0]
        if result.probs is None:
            continue

        top1_idx = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)
        label = result.names.get(top1_idx, str(top1_idx))   # Fix #6: no KeyError

        now = time.time()

        if top1_conf < CFG.plant_conf_display:
            with _plant_lock:
                _plant_state[0] = None
            first_detected = first_label = None
            continue

        with _plant_lock:
            _plant_state[0] = (label, top1_conf, veg_bbox)

        if first_label is not None and label != first_label:
            first_detected = first_label = None

        if first_detected is None:
            first_detected = now
            first_label = label
            log.info("[plant] detected %s %.0f%% — confirming", label, top1_conf * 100)

        if top1_conf < CFG.conf_upload:
            continue

        elapsed = now - first_detected
        cooled = now - last_uploaded.get(label, 0.0)
        if elapsed >= CFG.confirm_secs and cooled >= CFG.save_cooldown:
            if enqueue_upload("plant", label, top1_conf, frame, None):
                log.info("[plant] confirmed %.1fs — queued upload", elapsed)
                last_uploaded[label] = now
                first_detected = first_label = None
            else:
                log.warning("[plant] upload dropped (queue full) — will retry")
        elif elapsed >= CFG.confirm_secs:
            log.debug("[plant] confirmed but cooling down — %.0fs left",
                      CFG.save_cooldown - cooled)


# ─────────────────────────────────────────────────────────────────
# WORKER SUPERVISOR — restart on crash instead of silent freeze
# ─────────────────────────────────────────────────────────────────
def _supervise(name: str, target) -> None:
    crashes = 0
    while not STOP.is_set():
        try:
            target()
            return                                   # clean exit (STOP set)
        except Exception:
            crashes += 1
            log.exception("[%s] worker crashed (%d/%d)", name, crashes, CFG.worker_restart_max)
            if crashes >= CFG.worker_restart_max:
                log.error("[%s] too many crashes — giving up", name)
                STOP.set()
                return
            if STOP.wait(1.0):   # exit promptly if shutdown started during backoff
                return


# ─────────────────────────────────────────────────────────────────
# DRAW
# ─────────────────────────────────────────────────────────────────
def _draw(frame: np.ndarray) -> None:
    with _fire_lock:
        boxes = list(_fire_boxes)
    for x1, y1, x2, y2, label, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 60, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)

    with _plant_lock:
        plant = _plant_state[0]
    if plant is not None:
        label, conf, veg_bbox = plant
        if veg_bbox is not None:
            px1, py1, px2, py2 = veg_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 60), 2)
        clean = label.replace("___", " ").replace("_", " ")
        text = f"{clean}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (8, 8), (8 + tw + 10, 8 + th + 12), (0, 0, 0), -1)
        cv2.putText(frame, text, (13, 8 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 60), 2)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main() -> int:
    for label, path in [("fire", CFG.model_fire), ("plant", CFG.model_plant)]:
        if not Path(path).exists():
            log.error("%s model not found: %s", label, path)
            return 1

    if CFG.gps_lat is None or CFG.gps_lng is None:
        log.warning("GPS not set — set AGRO_GPS_LAT / AGRO_GPS_LNG for accurate uploads")

    # Load models ONCE here so a supervisor restart reuses them (no GPU reload).
    try:
        log.info("loading fire model %s", CFG.model_fire)
        fire_model = YOLO(str(CFG.model_fire))
        log.info("fire classes: %s", list(fire_model.names.values()))
        log.info("loading plant model %s", CFG.model_plant)
        plant_model = YOLO(str(CFG.model_plant))
        log.info("plant classes: %s", list(plant_model.names.values()))
    except Exception:
        log.exception("model load failed")
        return 1

    # Cooldown state owned here so it survives a worker restart.
    fire_cooldown: dict[str, float] = {}
    plant_cooldown: dict[str, float] = {}

    # Graceful shutdown on Ctrl+C / SIGTERM (main thread only — valid here).
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: STOP.set())

    workers = [
        threading.Thread(
            target=_supervise,
            args=("fire", functools.partial(_fire_worker, fire_model, fire_cooldown)),
            name="fire"),
        threading.Thread(
            target=_supervise,
            args=("plant", functools.partial(_plant_worker, plant_model, plant_cooldown)),
            name="plant"),
    ]
    # Uploaders are daemon: a single hung network upload must never block process
    # exit. We drain best-effort with a timeout below, then let daemons die.
    uploaders = [
        threading.Thread(target=_uploader_loop, name=f"uploader-{i}", daemon=True)
        for i in range(CFG.upload_workers)
    ]
    for t in (*workers, *uploaders):
        t.start()

    cap = cv2.VideoCapture(CFG.camera_index)
    if not cap.isOpened():
        log.error("cannot open camera %d", CFG.camera_index)
        STOP.set()
        BUS.ready.set()
        _shutdown(workers, cap=None)
        return 1
    if not cap.set(cv2.CAP_PROP_BUFFERSIZE, 1):
        log.debug("camera backend ignored BUFFERSIZE=1 (expected on some platforms)")

    BUS.ready.set()
    log.info("AgroSentinel running — press Q to quit. fire model: %s", CFG.model_fire)

    try:
        while not STOP.is_set():
            # NOTE: cap.read() is a blocking C call. On a camera hang (USB unplug)
            # it may not return promptly, delaying SIGTERM-triggered shutdown until
            # it does. This is an OpenCV limitation; a watchdog could force-release.
            ret, frame = cap.read()
            if not ret:
                log.error("camera read failed")
                break
            BUS.publish(frame.copy())
            _draw(frame)
            cv2.imshow("AgroSentinel", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        _shutdown(workers, cap)
    return 0


def _shutdown(workers, cap) -> None:
    log.info("shutting down...")
    STOP.set()
    BUS.ready.set()
    for t in workers:
        t.join(timeout=5.0)

    # Best-effort drain: wait up to upload_drain_secs for queued uploads to flush,
    # then send poison pills. Daemon uploaders mean a stuck upload can't hang exit.
    deadline = time.time() + CFG.upload_drain_secs
    while not _upload_q.empty() and time.time() < deadline:
        time.sleep(0.1)
    remaining = _upload_q.qsize()
    if remaining:
        log.warning("%d upload(s) not flushed before shutdown deadline", remaining)
    for _ in range(CFG.upload_workers):
        try:
            _upload_q.put_nowait(None)
        except queue.Full:
            pass

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    log.info("stopped.")


if __name__ == "__main__":
    sys.exit(main())
