# Hand Gesture + Face Control Room

Train a hand-gesture classifier on top of MediaPipe landmarks, enrich it with fine-grained hand analytics, and optionally add a face-expression model trained from external datasets. The local web UI stays the main testing surface.

## Workflow

1. Collect or import hand landmarks into the canonical CSV.
2. Train the hand classifier and review the generated evaluation artifacts.
3. Optionally train a face-expression classifier from an external dataset.
4. Run the browser dashboard to test live hand control, detailed hand signals, face pose, and face expressions in one place.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `mediapipe` does not install on `Python 3.14`, create the virtual environment with `Python 3.12` instead.

## Hand Collection

Collect manual hand samples:

```bash
python collect_data.py --labels peace,thumbs_up,fist,open_hand,point_down
```

Manual collection stores normalized hand landmarks in [data/gestures.csv](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/data/gestures.csv). Aim for at least `30+` samples per gesture per session across multiple sessions.

## Import External Hand Datasets

You do not need to label every gesture by webcam. Import a labeled image-folder dataset instead:

```bash
python import_hand_dataset.py --image-root external_datasets/hands
```

The importer expects a folder layout like:

```text
external_datasets/hands/
  peace/
    sample-001.jpg
  thumbs_up/
    sample-002.jpg
```

If an external dataset uses different label names, map them:

```bash
python import_hand_dataset.py \
  --image-root external_datasets/hands \
  --label-map '{"like":"thumbs_up","palm":"open_hand"}'
```

## Train The Hand Model

```bash
python train_model.py
```

This writes:

- [artifacts/gesture_model.joblib](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/artifacts/gesture_model.joblib)
- [artifacts/gesture_model_metrics.json](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/artifacts/gesture_model_metrics.json)
- [artifacts/confusion_matrix.png](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/artifacts/confusion_matrix.png)
- [artifacts/training_report.md](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/artifacts/training_report.md)

## Train The Face Model From External Datasets

The face-expression model is optional and designed for external datasets rather than manual browser labeling.

### FER2013 CSV

```bash
python train_face_model.py \
  --dataset external_datasets/fer2013/fer2013.csv \
  --dataset-format fer2013
```

### Generic Image Folders

```bash
python train_face_model.py \
  --dataset external_datasets/faces \
  --dataset-format image-folder
```

Expected image-folder layout:

```text
external_datasets/faces/
  happy/
    img-001.jpg
  neutral/
    img-002.jpg
```

This writes:

- [artifacts/face_expression_model.joblib](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/artifacts/face_expression_model.joblib)
- [artifacts/face_expression_metrics.json](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/artifacts/face_expression_metrics.json)
- [artifacts/face_expression_confusion_matrix.png](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/artifacts/face_expression_confusion_matrix.png)
- [artifacts/face_expression_report.md](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/artifacts/face_expression_report.md)

## Web UI

Run the local browser dashboard:

```bash
python web_ui.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The dashboard now shows:

- hand gesture collection and dry-run inference
- finger count, finger states, pinch target, pinch strength, and palm rotation
- face state, attention, head pose, smile, mouth-open, and blink signals
- face-expression prediction when a face model artifact exists
- separate hand and face training reports

The web UI always keeps hand actions in dry-run mode. Use [run_controller.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/run_controller.py) when you want real key presses.

## CLI Runtime

Dry-run the hand controller first:

```bash
python run_controller.py --dry-run
```

Then run it normally:

```bash
python run_controller.py
```

## Default Bindings

Default bindings live in [config/gesture_bindings.json](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/config/gesture_bindings.json):

- `peace` -> `space`
- `thumbs_up` -> `right`
- `fist` -> `escape`
- `open_hand` -> `up`
- `point_down` -> `down`

Supported action formats:

- `key:<name>`
- `hotkey:<k1>+<k2>+...`
- `print:<message>`
- `shell:<command>`
- `noop`

## Project Layout

- [collect_data.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/collect_data.py)
- [import_hand_dataset.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/import_hand_dataset.py)
- [train_model.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/train_model.py)
- [train_face_model.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/train_face_model.py)
- [web_ui.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/web_ui.py)
- [gesture_controller/landmarks.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/gesture_controller/landmarks.py)
- [gesture_controller/hand_analytics.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/gesture_controller/hand_analytics.py)
- [gesture_controller/face_landmarks.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/gesture_controller/face_landmarks.py)
- [gesture_controller/face_features.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/gesture_controller/face_features.py)
- [gesture_controller/face_modeling.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/gesture_controller/face_modeling.py)
- [gesture_controller/webapp.py](/Users/reon/Documents/Github%20Repos/hand-gesture-ml/hand-gesture/gesture_controller/webapp.py)
