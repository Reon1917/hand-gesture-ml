from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def bootstrap_local_venv(script_file: str, required_modules: tuple[str, ...]) -> None:
    missing = [module for module in required_modules if importlib.util.find_spec(module) is None]
    if not missing:
        return

    script_path = Path(script_file).resolve()
    project_root = script_path.parent
    venv_python = project_root / ".venv" / "bin" / "python"

    if venv_python.exists():
        current_executable = Path(sys.executable).absolute()
        target_executable = venv_python.absolute()
        if current_executable != target_executable:
            os.execv(str(target_executable), [str(target_executable), *sys.argv])

    missing_text = ", ".join(missing)
    raise SystemExit(
        f"Missing required Python modules: {missing_text}\n"
        "Activate the local virtualenv with `source .venv/bin/activate` and install dependencies with "
        "`python -m pip install -r requirements.txt`."
    )
