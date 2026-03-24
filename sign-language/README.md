# ASL Webcam Studio

Train a lightweight American Sign Language alphabet classifier from a Kaggle dataset, then run live webcam inference in the browser with built-in reference examples for non-signers.

The stack is intentionally CPU-friendly for an Apple Silicon laptop:

- classical vision features instead of a heavyweight GPU-only training stack
- FastAPI + vanilla JS for a minimal local web app
- reference cards generated from the same training dataset so you can copy the pose while testing

## Scope

This project is for **static ASL handshapes**, not full sentence-level sign language translation.

By default the training pipeline excludes:

- `J`
- `Z`
- `del`
- `nothing`
- `space`

`J` and `Z` are motion-based signs, so a single-frame image classifier is the wrong tool for them.

## Setup

```bash
cd "/Users/reon/Documents/Github Repos/hand-gesture-ml/sign-language"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Kaggle Credentials

Use either:

- `~/.kaggle/kaggle.json`, or
- `KAGGLE_USERNAME` and `KAGGLE_KEY`

The default dataset slug is [`grassknoted/asl-alphabet`](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

## Download The Dataset

```bash
python download_dataset.py
```

This creates a local link at `data/asl-alphabet` that points to the Kaggle download cache.

## Train

Balanced default for an M-series MacBook:

```bash
python train_model.py --download --max-images-per-class 1200
```

Useful options:

- `--include-motion` to keep `J` and `Z`
- `--include-special` to keep `space`, `del`, and `nothing`
- `--max-images-per-class 0` to use the full dataset
- `--quiet` to hide phase logging and progress bars

The CLI now prints dataset indexing, feature extraction progress bars, and per-model training updates.
If you accidentally run `python3 train_model.py` or `python3 web_ui.py` outside the active venv, the scripts will try to reuse the local `.venv` automatically.

Training writes:

- `artifacts/asl_model.joblib`
- `artifacts/metrics.json`
- `artifacts/training_report.md`
- `artifacts/confusion_matrix.png`
- `artifacts/examples/*.jpg`

## Run The Web App

```bash
python web_ui.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The UI includes:

- live webcam inference
- confidence-ranked predictions
- dataset reference cards beside the camera feed
- testing guidance for non-signers

## Testing Tips

- Use one hand and keep it centered in frame.
- Match the sample reference card exactly when possible.
- Use a neutral background and even light.
- Hold the pose still for a moment before changing to the next sign.

## Local Checks

```bash
python -m unittest discover -s tests
```
