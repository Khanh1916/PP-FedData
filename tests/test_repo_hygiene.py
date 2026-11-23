"""No tracked file breaks the rules of `.githooks/pre-commit` (no data, run outputs, data or model file types, files over 5 MB, local
working folders or personal agent files). The test runs the hook itself on the files, so the rules live in one place, and it also covers
a clone where the hook was never turned on."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-commit"


def _sh() -> str | None:
    """A POSIX shell: on PATH, or the one shipped with Git for Windows (next to `git --exec-path`)."""
    found = shutil.which("sh")
    if found:
        return found
    try:
        exec_path = Path(subprocess.run(["git", "--exec-path"], capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None
    for cand in (exec_path.parents[2] / "bin" / "sh.exe", exec_path.parents[2] / "usr" / "bin" / "sh.exe"):
        if cand.exists():
            return str(cand)
    return None


def _run_hook(paths: list[str]) -> subprocess.CompletedProcess:
    sh = _sh()
    if sh is None:
        pytest.skip("no POSIX shell to run the hook")
    return subprocess.run([sh, str(HOOK), *paths], capture_output=True, text=True, cwd=ROOT)


def _tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "-c", "core.quotepath=off", "ls-files"], capture_output=True, text=True, check=True, cwd=ROOT).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout: the tracked files are unknown")
    return [p for p in out.splitlines() if p]


def test_no_tracked_file_breaks_the_pre_commit_rules():
    files = _tracked()
    assert files
    for i in range(0, len(files), 100):                         # chunks keep the command line short on Windows
        r = _run_hook(files[i:i + 100])
        assert r.returncode == 0, r.stderr


def test_the_hook_refuses_data_outputs_and_local_files_and_lets_the_placeholders_through():
    bad = ["data/x.npz", "artifacts/run/model.pt", "results/runs.CSV", "configs/local.yaml", "Latex_FinalCapstone/main.tex", "CLAUDE.local.md",
           ".claude/settings.local.json", "notes/table.parquet"]
    ok = ["data/README.md", "artifacts/.gitkeep", "AGENTS.md"]
    r = _run_hook(bad + ok)
    assert r.returncode == 1
    for p in bad:
        assert f"refused {p} " in r.stderr, p
    for p in ok:
        assert f"refused {p} " not in r.stderr, p


def test_the_hook_refuses_a_file_over_the_size_limit(tmp_path):
    big, small = tmp_path / "big.txt", tmp_path / "small.txt"
    big.write_bytes(b"0" * 5_000_001)
    small.write_bytes(b"0" * 10)
    assert _run_hook([small.as_posix()]).returncode == 0
    r = _run_hook([big.as_posix(), small.as_posix()])
    assert r.returncode == 1 and "over the 5000000 byte limit" in r.stderr and small.as_posix() not in r.stderr
