"""Supervise the approved five-seed sensitivity grid and its validated handoff."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 1, 2, 3, 4)


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    import hashlib

    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run(args):
    directory = args.output_dir.resolve()
    source = args.source_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock_handle = (directory / "batch.lock").open("a")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = directory / "batch_status.json"
    state = {
        "status": "starting",
        "started_at_utc": now(),
        "supervisor_pid": os.getpid(),
        "seeds": list(SEEDS),
        "specifications_per_seed": 19,
        "B": 100,
        "tickers_per_draw": 200,
        "workers": 4,
        "source_directory": str(source),
        "completed_seeds": [],
    }
    save(state_path, state)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    environment.pop("FINEVAL_CHECKPOINT_DIR", None)
    expected = json.loads((ROOT / "results/run4/provenance/panel_digests.json").read_text())
    input_record = directory / "input_hashes.json"
    if input_record.exists() and json.loads(input_record.read_text()) != expected:
        raise ValueError("Batch input record differs from the run #4 identity")
    save(input_record, expected)
    reference = directory / "primary_reference.csv"
    if not reference.exists():
        shutil.copy2(ROOT / "results/run4/metric_summary.csv", reference)
    guarded = [
        "manuscript/sections/" + name + ".tex"
        for name in ("experiments", "appendix", "discussion", "conclusion")
    ]
    guard = directory / "manuscript_guard.json"
    if not guard.exists():
        save(guard, {name: digest(ROOT / name) for name in guarded})
    for seed in SEEDS:
        for name, wanted in expected.items():
            if digest(source / name) != wanted:
                raise ValueError(f"Panel identity changed before seed {seed}: {name}")
        folder = directory / f"seed_{seed}"
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.reproduce",
            "sensitivity",
            "--",
            "--fast",
            "--n-resamples",
            "100",
            "--tickers-per-draw",
            "200",
            "--n-jobs",
            "4",
            "--seed",
            str(seed),
            "--run-dir",
            str(folder),
        ]
        if seed == 42:
            command += ["--primary-reference", str(reference)]
        status = folder / "status.json"
        if status.exists() and json.loads(status.read_text()).get("status") == "completed":
            state["completed_seeds"].append(seed)
            continue
        succeeded = False
        for attempt in range(1, 4):
            run_command = command + (["--resume"] if (folder / "grid_spec.json").exists() else [])
            with (directory / f"seed_{seed}.log").open("ab") as log:
                child = subprocess.Popen(
                    run_command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
                state.update(
                    status="running",
                    current_seed=seed,
                    attempt=attempt,
                    benchmark_pid=child.pid,
                    command=run_command,
                    last_updated_at_utc=now(),
                )
                save(state_path, state)
                code = child.wait()
            if code == 0:
                succeeded = True
                break
            state.update(status="retrying", last_returncode=code, last_updated_at_utc=now())
            save(state_path, state)
            time.sleep(10)
        if not succeeded:
            raise RuntimeError(
                f"Seed {seed} failed after three attempts; completed specs preserved"
            )
        state["completed_seeds"].append(seed)
        save(state_path, state)
    state.update(status="validating_and_updating", last_updated_at_utc=now())
    save(state_path, state)
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (directory / "finalization.log").open("ab") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.finalize_sensitivity",
                "--run-dir",
                str(directory),
                "--workspace",
                str(ROOT),
            ],
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    state.update(status="completed", ended_at_utc=now(), last_updated_at_utc=now())
    save(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sensitivity")
    parser.add_argument("--source-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        path = args.output_dir.resolve() / "batch_status.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=now())
        save(path, state)
        raise


if __name__ == "__main__":
    main()
