from __future__ import annotations   # allow `dict | None` etc. on Python 3.9

import torch
import os
import json
import tempfile
import yaml
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Resolve paths relative to this script, not the current working directory,
# so train and detect agree on where training_runs/ lives regardless of cwd.
SCRIPT_DIR      = Path(__file__).resolve().parent
DATASET         = str(SCRIPT_DIR / "datasets" / "fire-8" / "data.yaml")

# ── Model — set PRETRAINED_MODEL to a downloaded fire/smoke .pt to skip scratch training
# Download from: https://universe.roboflow.com (search "fire smoke yolov8")
# Leave as None to start from generic ImageNet weights (slower, lower accuracy)
PRETRAINED_MODEL = None                 # e.g. "/path/to/fire_smoke_best.pt"
MODEL            = PRETRAINED_MODEL if PRETRAINED_MODEL else "yolov8m.pt"

# Fewer epochs needed when starting from a fire/smoke pretrained model
# FREEZE_EPOCHS must be > WARMUP_EPOCHS (5) or Phase 1 is 100% warmup with no stable training
FREEZE_EPOCHS   = 10 if PRETRAINED_MODEL else 15
FREEZE_LAYERS   = 10
FINETUNE_EPOCHS = 100 if PRETRAINED_MODEL else 300  # pretrained converges much faster

IMG_SIZE        = 1280                  # larger input catches small/distant fire
SEED            = 42                    # fixed seed for reproducible experiments


LR_PHASE1       = 0.001
LR_PHASE2       = 0.005
LRF             = 0.01                  # Final LR = lr0 * lrf (cosine decay)
MOMENTUM        = 0.937
WEIGHT_DECAY    = 0.0005
WARMUP_EPOCHS   = 5
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR  = 0.1
PATIENCE        = 50                    # more patience for long fine-tune

# ── Augmentation ───────────────────────────────────────────────
FLIPLR          = 0.5
FLIPUD          = 0.1                   # small chance — smoke direction varies
DEGREES         = 30.0                  # fire appears at any angle
TRANSLATE       = 0.1
SCALE           = 0.5
SHEAR           = 2.0
PERSPECTIVE     = 0.0001
HSV_H           = 0.015
HSV_S           = 0.9
HSV_V           = 0.6
MOSAIC          = 1.0
CLOSE_MOSAIC    = 75                    # keep mosaic diversity longer
MIXUP           = 0.3                   # stronger blending for smoke variation
COPY_PASTE      = 0.3                   # paste fire/smoke onto new backgrounds

# ── Loss weights ───────────────────────────────────────────────
BOX             = 7.5
CLS             = 1.0
DFL             = 1.5

# ── Save & Output ──────────────────────────────────────────────
PROJECT         = str(SCRIPT_DIR / "training_runs")
SAVE_PERIOD     = 10
CONF            = 0.10                  # lower threshold catches faint/early smoke

# ── Resume control ─────────────────────────────────────────────
# Set FORCE_RESTART = True to ignore saved state and train from scratch
FORCE_RESTART   = False  # set True only to wipe checkpoints and retrain from scratch

STATE_FILE      = os.path.join(PROJECT, "training_state.json")

# Guard: Phase 1 needs stable (non-warmup) epochs or the frozen-head phase is pointless.
assert FREEZE_EPOCHS > WARMUP_EPOCHS, (
    f"FREEZE_EPOCHS ({FREEZE_EPOCHS}) must exceed WARMUP_EPOCHS ({WARMUP_EPOCHS}); "
    f"otherwise Phase 1 is 100% warmup with no stable training."
)


# ╔══════════════════════════════════════════════════════════════╗
# ║                     DEVICE DETECTION                         ║
# ╚══════════════════════════════════════════════════════════════╝

def detect_device():
  
    if not torch.cuda.is_available():
        print("=" * 60)
        print("  ❌  ERROR: No CUDA GPU detected on this server!")
        print("=" * 60)
        print("  Make sure CUDA drivers and PyTorch CUDA are installed.")
        print("  Check with: python -c \"import torch; print(torch.cuda.is_available())\"")
        print("  Install:    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        raise SystemExit(1)

    gpu_count = torch.cuda.device_count()
    name      = torch.cuda.get_device_name(0)
    # Use the SMALLEST GPU's VRAM on heterogeneous multi-GPU rigs so the shared
    # batch size never OOMs the weakest card.
    vram      = min(
        torch.cuda.get_device_properties(i).total_memory
        for i in range(gpu_count)
    ) / 1e9

    print(f"  GPUs found : {gpu_count}x {name}")
    print(f"  VRAM (each): {vram:.1f} GB")

    # Multi-GPU: use all available GPUs
    if gpu_count > 1:
        device = ",".join(str(i) for i in range(gpu_count))
        print(f"  Mode       : Multi-GPU ({device})")
    else:
        device = "0"
        print(f"  Mode       : Single GPU")

    # Workers: scale with GPU tier
    if   vram >= 40: workers = 16   # A100 / H100
    elif vram >= 20: workers = 12   # RTX 3090/4090
    elif vram >= 16: workers = 8    # RTX 3080/4080 / V100
    elif vram >= 10: workers = 6
    elif vram >=  6: workers = 4
    else:            workers = 2

    # AMP: safe on modern GPUs, problematic on older consumer cards
    amp = True
    if "1650" in name or "1660" in name or "1050" in name or "1060" in name:
        amp = False
        print(f"  AMP        : Disabled (older GPU — avoids NaN losses)")
    else:
        print(f"  AMP        : Enabled (faster training with mixed precision)")

    return device, int(vram), workers, amp, gpu_count


def auto_batch(vram_gb, gpu_count):
    
    if   vram_gb >= 40: batch_per_gpu = 64
    elif vram_gb >= 24: batch_per_gpu = 32
    elif vram_gb >= 16: batch_per_gpu = 16
    elif vram_gb >= 10: batch_per_gpu = 12
    elif vram_gb >=  6: batch_per_gpu = 8
    else:               batch_per_gpu = 1

    total_batch = batch_per_gpu * gpu_count
    print(f"  Batch      : {batch_per_gpu}/GPU × {gpu_count} GPU = {total_batch} total")
    return total_batch



def shared_args(device, batch, workers, amp):
    return dict(
        data            = DATASET,
        imgsz           = IMG_SIZE,
        batch           = batch,
        device          = device,
        workers         = workers,
        momentum        = MOMENTUM,
        weight_decay    = WEIGHT_DECAY,
        warmup_epochs   = WARMUP_EPOCHS,
        warmup_momentum = WARMUP_MOMENTUM,
        warmup_bias_lr  = WARMUP_BIAS_LR,
        patience        = PATIENCE,
        fliplr          = FLIPLR,
        flipud          = FLIPUD,
        degrees         = DEGREES,
        translate       = TRANSLATE,
        scale           = SCALE,
        shear           = SHEAR,
        perspective     = PERSPECTIVE,
        hsv_h           = HSV_H,
        hsv_s           = HSV_S,
        hsv_v           = HSV_V,
        mosaic          = MOSAIC,
        close_mosaic    = CLOSE_MOSAIC,
        mixup           = MIXUP,
        copy_paste      = COPY_PASTE,
        box             = BOX,
        cls             = CLS,
        dfl             = DFL,
        cache           = "disk",      # Disk cache — safe for large datasets (use "ram" only if RAM > 64GB)
        amp             = amp,
        cos_lr          = True,         # Cosine LR schedule — better for long training
        multi_scale     = True,         # random resize each batch — robust to fire at any scale
        label_smoothing = 0.1,          # prevent overconfidence on fuzzy fire/smoke boundaries
        seed            = SEED,
        plots           = True,
        verbose         = True,
        save_period     = SAVE_PERIOD,
        project         = PROJECT,
    )



def save_state(state: dict):
    """Atomic write: dump to a temp file in the same dir, then os.replace().

    Prevents a half-written training_state.json if the process is killed
    mid-save (OOM / power loss), which would otherwise crash the next run.
    """
    os.makedirs(PROJECT, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=PROJECT, prefix=".state-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)   # atomic on POSIX/NTFS
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

def load_state() -> dict | None:
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ⚠️  Corrupt/unreadable state file ({exc}) — starting fresh.")
        try:
            os.remove(STATE_FILE)
        except OSError:
            pass
        return None

def detect_resume_state():
    if FORCE_RESTART:
        print("  FORCE_RESTART=True — ignoring any saved checkpoints.")
        return _fresh_state()

    state = load_state()
    if state is None:
        return _fresh_state()

    run_name   = state["run_name"]
    phase1_dir = state["phase1_dir"]
    phase2_dir = state["phase2_dir"]

    if state.get("phase2_done"):
        return dict(mode="done", run_name=run_name,
                    phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                    p1_last=None, p2_last=None, phase1_best=None)

    if state.get("phase1_done"):
        phase1_best = state.get("phase1_best")
        p2_last = os.path.join(phase2_dir, "weights", "last.pt")
        if os.path.exists(p2_last):
            return dict(mode="resume_phase2", run_name=run_name,
                        phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                        p1_last=None, p2_last=p2_last, phase1_best=phase1_best)
        else:
            return dict(mode="skip_to_phase2", run_name=run_name,
                        phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                        p1_last=None, p2_last=None, phase1_best=phase1_best)

    p1_last = os.path.join(phase1_dir, "weights", "last.pt")
    if os.path.exists(p1_last):
        return dict(mode="resume_phase1", run_name=run_name,
                    phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                    p1_last=p1_last, p2_last=None, phase1_best=None)

    print("  State file found but no weights on disk — starting fresh.")
    return _fresh_state()

def _fresh_state():
    # Remove stale state so a later non-forced run doesn't pick up old weights
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    run_name   = f"fire_{datetime.now().strftime('%Y%m%d_%H%M')}"
    phase1_dir = os.path.join(PROJECT, run_name + "_phase1")
    phase2_dir = os.path.join(PROJECT, run_name + "_phase2")
    return dict(mode="fresh", run_name=run_name,
                phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                p1_last=None, p2_last=None, phase1_best=None)



def validate(model, output_dir):
    print("\n  Validating best model...")
    best = os.path.join(output_dir, "weights", "best.pt")
    if os.path.exists(best):
        model = YOLO(best)

    # Pass data= explicitly: a reloaded best.pt validates against the path baked
    # into its weights otherwise, which may be stale or wrong.
    metrics = model.val(data=DATASET, conf=CONF, augment=True, iou=0.4)  # TTA + lower IoU separates overlapping fire+smoke
    map50   = metrics.box.map50
    map5095 = metrics.box.map
    prec    = metrics.box.mp or 0.0
    rec     = metrics.box.mr or 0.0
    f1      = 2 * prec * rec / (prec + rec + 1e-9)

    g = lambda v, t: "✅" if v >= t else ("⚠️ " if v >= t * 0.85 else "❌")

    print("\n" + "=" * 52)
    print("  FINAL RESULTS")
    print("=" * 52)
    print(f"  {g(map50,  0.85)}  mAP50     : {map50:.4f}   (target ≥ 0.85)")
    print(f"  {g(map5095,0.65)}  mAP50-95  : {map5095:.4f}   (target ≥ 0.65)")
    print(f"  {g(prec,   0.80)}  Precision : {prec:.4f}   (target ≥ 0.80)")
    print(f"  {g(rec,    0.80)}  Recall    : {rec:.4f}   (target ≥ 0.80)")
    print(f"       F1-Score  : {f1:.4f}")
    print("=" * 52)

    if map50 >= 0.85:
        print("\n  ✅ Excellent! Model is ready for deployment.")
    elif map50 >= 0.70:
        print("\n  ⚠️  Good model. Consider more epochs or more data.")
    else:
        print("\n  ❌ Low accuracy. Try more epochs or a larger dataset.")

    return metrics



def main():
    print("=" * 60)
    print("   YOLOv8x Fire & Smoke — Server-Grade Training")
    print("=" * 60)

    # ── Detect GPU ────────────────────────────────────────────
    device, vram, workers, amp, gpu_count = detect_device()
    batch = auto_batch(vram, gpu_count)
    os.makedirs(PROJECT, exist_ok=True)

    rs = detect_resume_state()
    mode       = rs["mode"]
    run_name   = rs["run_name"]
    phase1_dir = rs["phase1_dir"]
    phase2_dir = rs["phase2_dir"]

    print(f"\n  Model      : {MODEL}")
    print(f"  Strategy   : STAGED (Phase 1: frozen backbone → Phase 2: full fine-tune)")
    print(f"  Phase 1    : {FREEZE_EPOCHS} epochs — AdamW, frozen backbone (head warmup)")
    print(f"  Phase 2    : {FINETUNE_EPOCHS} epochs — SGD + cosine LR, all layers + TTA validation")
    print(f"  Image size : {IMG_SIZE}px")
    print(f"  Seed       : {SEED}")
    print(f"  Run name   : {run_name}")
    print(f"  Resume mode: {mode}")

    # ── Class imbalance warning ────────────────────────────────
    try:
        with open(DATASET) as f:
            d = yaml.safe_load(f)
        print(f"\n  Classes    : {d.get('names', 'unknown')}")
        print(f"  ⚠️  Verify train/labels has balanced fire vs smoke counts before training.")
    except Exception:
        pass

    # ── Pretrained model compatibility check ──────────────────
    if PRETRAINED_MODEL:
        print(f"\n  Pretrained : {PRETRAINED_MODEL}")
        try:
            probe = YOLO(PRETRAINED_MODEL)
            model_classes = list(probe.names.values())
            expected      = ["fire", "smoke"]
            if model_classes == expected:
                print(f"  ✅ Class check passed: {model_classes}")
            else:
                print(f"  ⚠️  Class mismatch: model has {model_classes}, expected {expected}")
                print(f"     Phase 1 will retrain the detection head — this is safe to continue.")
            del probe
        except Exception as e:
            print(f"  ❌ Could not load pretrained model: {e}")
            raise SystemExit(1)

    # ── Already finished ───────────────────────────────────────
    if mode == "done":
        print("\n  Training already complete for this run.")
        print(f"  Results in: {phase2_dir}/")
        print("  Set FORCE_RESTART=True to retrain from scratch.")
        validate(YOLO(os.path.join(phase2_dir, "weights", "best.pt")), phase2_dir)
        raise SystemExit(0)

    
    if mode in ("fresh", "resume_phase1"):
        print("\n" + "─" * 60)
        if mode == "resume_phase1":
            print(f"  RESUMING PHASE 1 from: {rs['p1_last']}")
        else:
            print(f"  PHASE 1 / 2 — Frozen backbone ({FREEZE_EPOCHS} epochs, AdamW)")
            print(f"  Trains detection head only — fast convergence on new classes")
        print("─" * 60 + "\n")

        phase1_name = run_name + "_phase1"
        torch.cuda.empty_cache()

        if mode == "resume_phase1":
            model = YOLO(rs["p1_last"])
            model.train(
                **shared_args(device, batch, workers, amp),
                resume    = True,
                epochs    = FREEZE_EPOCHS,
                freeze    = FREEZE_LAYERS,
                optimizer = "AdamW",
                lr0       = LR_PHASE1,
                lrf       = LRF,
                name      = phase1_name,
            )
        else:
            model = YOLO(MODEL)
            model.train(
                **shared_args(device, batch, workers, amp),
                epochs    = FREEZE_EPOCHS,
                freeze    = FREEZE_LAYERS,
                optimizer = "AdamW",
                lr0       = LR_PHASE1,
                lrf       = LRF,
                name      = phase1_name,
            )

        # ── Capture REAL paths from trainer ───────────────────
        real_phase1_dir  = str(model.trainer.save_dir)
        real_phase1_best = str(model.trainer.best)

        # Fallback to last.pt if best.pt doesn't exist
        if not os.path.exists(real_phase1_best):
            real_phase1_best = str(model.trainer.last)
            print(f"  ⚠️  best.pt missing, falling back to last.pt")

        print(f"\n  ✅ Phase 1 complete")
        print(f"     Dir  : {real_phase1_dir}")
        print(f"     Best : {real_phase1_best}")

        save_state(dict(
            run_name    = run_name,
            phase1_dir  = real_phase1_dir,
            phase1_best = real_phase1_best,
            phase2_dir  = phase2_dir,
            phase1_done = True,
            phase2_done = False,
        ))

        phase1_dir     = real_phase1_dir
        p1_best_for_p2 = real_phase1_best

    else:
        model          = None
        phase1_dir     = rs["phase1_dir"]   # ensure phase1_dir is always set from state
        p1_best_for_p2 = rs["phase1_best"]
        # Guard: old/partial state files may lack phase1_best — fail clearly
        # instead of crashing later on os.path.exists(None).
        if not p1_best_for_p2:
            print("  ❌ ERROR: state file has no Phase 1 weights path "
                  "(corrupt or pre-upgrade state). Set FORCE_RESTART=True to retrain.")
            raise SystemExit(1)

    
    print("\n" + "─" * 60)
    if mode == "resume_phase2":
        print(f"  RESUMING PHASE 2 from: {rs['p2_last']}")
    else:
        print(f"  PHASE 2 / 2 — Full fine-tune ({FINETUNE_EPOCHS} epochs, SGD + cosine LR)")
        print(f"  All layers unlocked — deep feature adaptation")
        print(f"  Loading weights: {p1_best_for_p2}")
    print("─" * 60 + "\n")

    
    if mode != "resume_phase2":
        if not os.path.exists(p1_best_for_p2):
            print(f"  ❌ ERROR: Phase 1 weights not found: {p1_best_for_p2}")
            print(f"  Check your {PROJECT}/ folder and update the path manually.")
            raise SystemExit(1)
        print(f"  ✅ Phase 1 weights verified on disk")

    phase2_name = run_name + "_phase2"

    if mode == "resume_phase2":
        model2 = YOLO(rs["p2_last"])
        model2.train(
            **shared_args(device, batch, workers, amp),
            resume    = True,
            epochs    = FINETUNE_EPOCHS,
            freeze    = 0,
            optimizer = "SGD",
            lr0       = LR_PHASE2,
            lrf       = LRF,
            name      = phase2_name,
        )
    else:
        model2 = YOLO(p1_best_for_p2)
        model2.train(
            **shared_args(device, batch, workers, amp),
            epochs    = FINETUNE_EPOCHS,
            freeze    = 0,
            optimizer = "SGD",
            lr0       = LR_PHASE2,
            lrf       = LRF,
            name      = phase2_name,
        )

    real_phase2_dir = str(model2.trainer.save_dir)

    save_state(dict(
        run_name    = run_name,
        phase1_dir  = phase1_dir,
        phase1_best = p1_best_for_p2,
        phase2_dir  = real_phase2_dir,
        phase1_done = True,
        phase2_done = True,
    ))

    # ── Validate & summarize ───────────────────────────────────
    validate(model2, real_phase2_dir)

    best_final = os.path.join(real_phase2_dir, "weights", "best.pt")
    print(f"\n  ✅ Training complete! Files saved to: {real_phase2_dir}/")
    print(f"   └── weights/best.pt       ← use this for inference")
    print(f"   └── weights/last.pt")
    print(f"   └── results.png           ← loss & mAP curves")
    print(f"   └── confusion_matrix.png  ← per-class accuracy")

    # ── Export to ONNX for edge/CPU deployment ─────────────────
    onnx_path = str(Path(best_final).with_suffix(".onnx"))
    print(f"\n  Exporting best model to ONNX...")
    try:
        export_model = YOLO(best_final)
        export_model.export(format="onnx", imgsz=IMG_SIZE, dynamic=True)
        print(f"  ✅ ONNX model saved: {onnx_path}")
        print(f"     Use for edge cameras, Jetson, or CPU-only deployments.")
    except Exception as e:
        print(f"  ⚠️  ONNX export failed: {e} — best.pt still usable.")

    print(f"\n  Inference command (PyTorch):")
    print(f'   yolo detect predict model="{os.path.abspath(best_final)}" source=your_image.jpg conf=0.10 iou=0.4')
    print(f"\n  Inference command (ONNX):")
    print(f'   yolo detect predict model="{os.path.abspath(onnx_path)}" source=your_image.jpg conf=0.10 iou=0.4')


if __name__ == '__main__':
    main()