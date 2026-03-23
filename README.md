# Hand Gesture Controller

Train a gesture classifier on top of MediaPipe hand landmarks, then map live predictions to keyboard shortcuts or shell commands.

## Workflow

1. Collect landmark samples from your webcam into a CSV.
2. Train a classifier on the saved gesture labels.
3. Run live inference and trigger actions when a gesture is stable.

## Setup

Create a virtual environment inside the repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `mediapipe` fails to install on `Python 3.14`, use a `Python 3.12` interpreter for the virtual environment instead:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MediaPipe support tends to lag the newest Python release, so `3.12` is the safer path on macOS when wheels are not yet published for `3.14`.

## Collect Data

Start with five labels:

```bash
python collect_data.py --labels peace,thumbs_up,fist,open_hand,point_down
```

On the first run with the MediaPipe Tasks backend, the project will download the hand-landmarker model into [models/hand_landmarker.task](/Users/reon/Documents/Github Repos/hand-gesture-ml/models/hand_landmarker.task).

Collection mode does not recognize gestures yet. It only tracks your hand landmarks and saves the current frame when you press the matching number key.

Controls:

- `1-9`: save the current hand landmarks for the matching label
- `q`: quit

Aim for at least `100` samples per gesture with small variations in angle, distance, and hand position.

The collector stores normalized landmarks in [data/gestures.csv](/Users/reon/Documents/Github Repos/hand-gesture-ml/data/gestures.csv).

## Train

```bash
python train_model.py
```

This writes:

- [artifacts/gesture_model.joblib](/Users/reon/Documents/Github Repos/hand-gesture-ml/artifacts/gesture_model.joblib)
- [artifacts/gesture_model_metrics.json](/Users/reon/Documents/Github Repos/hand-gesture-ml/artifacts/gesture_model_metrics.json)

## Run Live Inference

Dry run first so you can confirm predictions without sending key presses:

```bash
python run_controller.py --dry-run
```

Then run it normally:

```bash
python run_controller.py
```

Default bindings live in [config/gesture_bindings.json](/Users/reon/Documents/Github Repos/hand-gesture-ml/config/gesture_bindings.json):

- `peace` -> `space`
- `thumbs_up` -> `right`
- `fist` -> `escape`
- `open_hand` -> `up`
- `point_down` -> `down`

These defaults are tuned for slides, YouTube, and similar keyboard-driven apps.

## Action Bindings

Supported action formats:

- `key:<name>` for a single key press
- `hotkey:<k1>+<k2>+...` for multi-key shortcuts
- `print:<message>` for console-only debugging
- `shell:<command>` to launch a local shell command
- `noop` to ignore a gesture

Example:

```json
{
  "peace": "key:space",
  "thumbs_up": "hotkey:command+right",
  "fist": "key:escape",
  "open_hand": "key:up",
  "point_down": "shell:osascript -e 'beep'"
}
```

## macOS Notes

- Grant Accessibility permissions to the terminal app you use for live control, otherwise simulated key presses will be blocked.
- The detector is configured with `model_complexity=1`, which is a good default for real-time tracking on Apple Silicon.
- If your installed `mediapipe` package only exposes `tasks` and not `solutions`, this project handles that automatically and uses the Tasks hand-landmarker backend instead.

## Project Layout

- [collect_data.py](/Users/reon/Documents/Github Repos/hand-gesture-ml/collect_data.py)
- [train_model.py](/Users/reon/Documents/Github Repos/hand-gesture-ml/train_model.py)
- [run_controller.py](/Users/reon/Documents/Github Repos/hand-gesture-ml/run_controller.py)
- [gesture_controller/landmarks.py](/Users/reon/Documents/Github Repos/hand-gesture-ml/gesture_controller/landmarks.py)
- [gesture_controller/modeling.py](/Users/reon/Documents/Github Repos/hand-gesture-ml/gesture_controller/modeling.py)
- [gesture_controller/actions.py](/Users/reon/Documents/Github Repos/hand-gesture-ml/gesture_controller/actions.py)
