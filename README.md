# Fire & Smoke Detection

A YOLOv8-based pipeline for training and running real-time **fire** and **smoke**
detection models. Built for edge deployment: the training script auto-tunes to the
available GPU, and the inference script runs supervised background workers with a
graceful, drain-on-shutdown upload path.

## Features

- **Two-phase transfer learning** — a frozen-backbone warmup phase followed by a
  full fine-tune, with resume-from-checkpoint support.
- **Hardware-aware training** — device, batch size, dataloader workers, and mixed
  precision are auto-detected from the installed GPU(s) (`train_fire.py`).
- **High small-object recall** — 1280px input, aggressive augmentation, and a low
  confidence threshold tuned to catch faint or distant smoke.
- **Production inference** — background detection workers, bounded upload queue,
  worker supervision/restart, and graceful shutdown (`detect.py`).
- **Fully env-configurable** — inference behavior is overridable via `AGRO_*`
  environment variables, so the same binary runs in dev and on-device.

## Repository layout

| Path               | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| `train_fire.py`    | Two-phase YOLOv8 training pipeline with auto hardware tuning.    |
| `detect.py`        | Real-time inference: fire detection + plant classification.     |
| `data.yaml`        | Dataset config (classes: `fire`, `smoke`).                      |
| `requirements.txt` | Python dependencies.                                             |
| `.gitignore`       | Excludes weights, datasets, and training outputs.               |

> **Note:** `detect.py` imports a `detect_and_upload` module (the Firebase upload
> handlers) that must be supplied for the inference pipeline to run end-to-end.

## Requirements

- Python 3.9+
- A CUDA-capable GPU (required for training)

```bash
pip install -r requirements.txt
```

## Dataset

Place your dataset under `datasets/fire-8/` in standard YOLO format and point
`data.yaml` at it (the default `path: .` resolves relative to the config file):

```
datasets/fire-8/
├── train/images  train/labels
├── valid/images  valid/labels
└── test/images   test/labels
```

Classes are defined in `data.yaml`:

```yaml
nc: 2
names: ['fire', 'smoke']
```

You can source a labelled fire/smoke dataset from
[Roboflow Universe](https://universe.roboflow.com) (search "fire smoke yolov8").
To start from fire/smoke pretrained weights instead of generic ImageNet weights,
set `PRETRAINED_MODEL` at the top of `train_fire.py`.

## Training

```bash
python train_fire.py
```

The script:

1. Detects the GPU(s) and picks batch size, workers, and AMP automatically.
2. **Phase 1** — trains with the backbone frozen (fast, stable head warmup).
3. **Phase 2** — unfreezes and fine-tunes the whole network.
4. Checkpoints to `training_runs/` and resumes automatically if interrupted.
   Set `FORCE_RESTART = True` to wipe state and train from scratch.

Best weights land in `training_runs/<run>_phase2*/weights/best.pt`.

## Inference

```bash
python detect.py
```

`detect.py` auto-detects the newest trained model under `training_runs/` and
falls back to `models/fire_best.pt`. Common overrides:

| Variable            | Default | Description                              |
| ------------------- | ------- | ---------------------------------------- |
| `AGRO_CAMERA`       | `0`     | Camera index.                            |
| `AGRO_FIRE_CONF`    | `0.35`  | Display confidence threshold.            |
| `AGRO_CONF_UPLOAD`  | `0.80`  | Confidence required to upload an event.  |
| `AGRO_FIRE_IMGSZ`   | `1280`  | Inference image size.                    |
| `AGRO_GPS_LAT/LNG`  | unset   | GPS coordinates attached to uploads.     |
| `AGRO_LOG_LEVEL`    | `INFO`  | Logging verbosity.                       |

Press `Q` (or `Ctrl+C`) to shut down; workers are joined and the upload queue is
drained before exit.

## License

Released under the [MIT License](LICENSE).
