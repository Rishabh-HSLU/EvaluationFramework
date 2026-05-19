"""
Tiny shared run-logging helper.

create_run_dir(task) -> runs/<UTC-timestamp>_<task>/  with files:
    config.yaml  (params + seeds)
    log.txt      (captured stdout)
    env.txt      (uv pip list)
    results.csv  (written by the caller)
"""

from __future__ import annotations

import datetime as dt
import io
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def create_run_dir(task: str) -> Path:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"{ts}_{task}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config(run_dir: Path, config: dict[str, Any]) -> None:
    """Plain YAML — flat dict only (no dep on PyYAML)."""
    lines = []
    for k, v in config.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {_yaml_value(vv)}")
        elif isinstance(v, list):
            lines.append(f"{k}: {v!r}")
        else:
            lines.append(f"{k}: {_yaml_value(v)}")
    (run_dir / "config.yaml").write_text("\n".join(lines) + "\n")


def _yaml_value(v: Any) -> str:
    if isinstance(v, str):
        return v
    return repr(v)


def write_env(run_dir: Path) -> None:
    try:
        out = subprocess.check_output(
            ["uv", "pip", "list"],
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        out = f"uv pip list unavailable: {e}\n"
    (run_dir / "env.txt").write_text(out)


class _Tee(io.TextIOBase):
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


@contextmanager
def tee_stdout(log_path: Path):
    fh = log_path.open("w")
    orig = sys.stdout
    sys.stdout = _Tee(orig, fh)
    try:
        yield
    finally:
        sys.stdout = orig
        fh.close()
