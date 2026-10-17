"""Launch reproduction steps with numerical settings applied before imports."""

from __future__ import annotations

import argparse
import os
import platform
import sys

MODULES = {
    "curate": "scripts.build_curated_datasets",
    "controls": "scripts.baseline_generation",
    "sbbts-panel": "scripts.sbbts_panel",
    "sbbts-train": "scripts.sbbts_train",
    "sbbts-generate": "scripts.sbbts_generate",
    "benchmark": "scripts.run_benchmark",
    "profiles": "scripts.plot_metric_profiles",
    "sensitivity": "fineval.scripts.sensitivity.run_sweep",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproduction step with the tested numerical environment. "
            "Pass step-specific options after --."
        )
    )
    parser.add_argument("step", choices=MODULES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    environment = os.environ.copy()
    if platform.machine().lower() in {"x86_64", "amd64"}:
        # Match the tested local CPU profile instead of selecting AVX-512 kernels.
        disabled = environment.get("NPY_DISABLE_CPU_FEATURES", "").split(",")
        disabled.extend(["X86_V4", "AVX512_ICL", "AVX512_SPR"])
        environment["NPY_DISABLE_CPU_FEATURES"] = ",".join(
            dict.fromkeys(feature.strip() for feature in disabled if feature.strip())
        )
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    # A new interpreter is essential: setting these after NumPy imports is too late.
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", MODULES[args.step], *arguments],
        environment,
    )


if __name__ == "__main__":
    main()
