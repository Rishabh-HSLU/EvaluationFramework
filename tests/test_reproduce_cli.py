"""The launcher must configure the child before scientific imports happen."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_reproduction_child_receives_profile_arguments_and_preserves_exit(tmp_path):
    probe = tmp_path / "reproduction_probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps({\n"
        " 'numpy_loaded': 'numpy' in sys.modules,\n"
        " 'disabled': os.environ.get('NPY_DISABLE_CPU_FEATURES', ''),\n"
        " 'threads': [os.environ.get(k) for k in "
        "('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')],\n"
        " 'arguments': sys.argv[2:]}))\n"
        "sys.exit(17)\n"
    )
    output = tmp_path / "probe.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(Path(__file__).resolve().parents[1])]
    )
    environment["OPENBLAS_NUM_THREADS"] = "8"
    environment["NPY_DISABLE_CPU_FEATURES"] = "AVX512_ICL"
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.reproduce import MODULES, main; "
            "MODULES['controls'] = 'reproduction_probe'; "
            "main(['controls', '--', * __import__('sys').argv[1:]])",
            str(output),
            "--seed",
            "42",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 17, process.stderr
    result = json.loads(output.read_text())
    assert result["numpy_loaded"] is False
    assert result["threads"] == ["1", "1", "1"]
    assert result["arguments"] == ["--seed", "42"]
    import platform

    if platform.machine().lower() in {"x86_64", "amd64"}:
        assert set(result["disabled"].split(",")) == {"X86_V4", "AVX512_ICL", "AVX512_SPR"}
