from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pyautogui
except Exception:  # pragma: no cover - runtime dependency on GUI access
    pyautogui = None
else:
    pyautogui.FAILSAFE = False


def load_bindings(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Bindings file must contain a JSON object.")
    return {str(key): str(value) for key, value in data.items()}


@dataclass(slots=True)
class ActionExecutor:
    dry_run: bool = False
    cooldown_seconds: float = 1.0
    _last_triggered: dict[str, float] = field(default_factory=dict)
    last_message: str = "Waiting for a stable gesture"

    def maybe_execute(self, gesture: str, action_spec: str) -> bool:
        now = time.monotonic()
        if now - self._last_triggered.get(gesture, 0.0) < self.cooldown_seconds:
            return False

        self._last_triggered[gesture] = now
        self.last_message = f"{gesture} -> {action_spec}"
        if self.dry_run:
            print(self.last_message)
            return True

        perform_action(action_spec)
        return True


def perform_action(action_spec: str) -> None:
    if action_spec == "noop":
        return

    if action_spec.startswith("print:"):
        print(action_spec.removeprefix("print:"))
        return

    if action_spec.startswith("shell:"):
        subprocess.Popen(action_spec.removeprefix("shell:"), shell=True)
        return

    if pyautogui is None:
        raise RuntimeError(
            "pyautogui is not available. Install requirements and grant Accessibility access on macOS."
        )

    if action_spec.startswith("key:"):
        pyautogui.press(action_spec.removeprefix("key:"))
        return

    if action_spec.startswith("hotkey:"):
        keys = [part.strip() for part in action_spec.removeprefix("hotkey:").split("+") if part.strip()]
        if not keys:
            raise ValueError(f"Invalid hotkey action: {action_spec}")
        pyautogui.hotkey(*keys)
        return

    raise ValueError(f"Unsupported action spec: {action_spec}")
