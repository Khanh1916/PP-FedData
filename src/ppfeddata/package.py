"""Phase 13: the reproducibility bundle (`ppfeddata package`).

Spec: "save `pip freeze`, `configs/`, `split_manifest.json`, `feature_schema.json` together with the results". Written to `results/repro/`:
- `pip_freeze.txt`    the installed libraries (the same text is written to `requirements-lock.txt`, which is what the README tells a reader to install);
- `environment.json`  Python, platform, versions of the libraries the study depends on, git commit and whether the tree was clean, label mode and seeds;
- `configs/`          a copy of `configs/*.yaml` (not `local.yaml`, which holds the machine's raw-data path) and `configs/exp/*.yaml`;
- `MANIFEST.json`     SHA-256 and size of the ledger, summary, interpretation, final report, the split manifest, the feature schema, the configs and the lock file.
The feature schema is also copied to `results/manifests/feature_schema_<mode>.json` next to the split manifest (parameters of the encoding and counts; no data row), so the
README, the report and the demo can use it on a machine without `data/`.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from ppfeddata.utils import config_hash, git_commit

KEY_LIBS = ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "imbalanced-learn", "torch", "flwr", "opacus", "ray", "optuna", "joblib", "matplotlib", "pyyaml", "streamlit",
            "pytest", "psutil")
ROOT = Path(__file__).resolve().parents[2]
EDITABLE_NOTE = "# ppfeddata (this repository) is installed with `pip install -e .`"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pip_freeze() -> list[str]:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def sanitize_freeze(lines: list[str]) -> list[str]:
    """`pip freeze` of an editable install shows a VCS URL or a local path of the machine: replace it, in place, with a comment (the package is this repository).
    The order of pip is kept."""
    out, seen = [], False
    for ln in lines:
        if ln.startswith("-e ") or "#egg=ppfeddata" in ln or ln.lower().startswith("ppfeddata @") or ln.lower().startswith("ppfeddata=="):
            if not seen:
                out.append(EDITABLE_NOTE)
                seen = True
            continue
        out.append(ln)
    return out


def lib_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in KEY_LIBS:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def git_dirty(root: Path = ROOT) -> bool | None:
    try:
        return bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=root).stdout.strip())
    except Exception:
        return None


def environment(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "libraries": lib_versions(),
            "git_commit": git_commit(), "git_dirty": git_dirty(), "config_hash": config_hash(cfg), "label_mode": cfg["label_mode"], "seeds": list(cfg["seeds"])}


def copy_configs(dst: Path, root: Path = ROOT) -> list[str]:
    """`configs/*.yaml` except local.yaml, and configs/exp/*.yaml, under `dst/configs`."""
    out = []
    for src in sorted((root / "configs").glob("*.yaml")) + sorted((root / "configs" / "exp").glob("*.yaml")):
        if src.name == "local.yaml":
            continue
        rel = src.relative_to(root / "configs")
        (dst / "configs" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst / "configs" / rel)
        out.append(str(Path("configs") / rel).replace("\\", "/"))
    return out


def _rel(p: Path, root: Path) -> str:
    """Path as written in the manifest: relative to the repository (never the absolute path of this machine) and with forward slashes."""
    p = Path(p)
    try:
        p = p.resolve().relative_to(Path(root).resolve())
    except ValueError:
        pass
    return str(p).replace("\\", "/")


def package(cfg: dict[str, Any], out_dir: str | Path | None = None, update_lock: bool = True, root: Path = ROOT, freeze: list[str] | None = None) -> dict[str, Any]:
    """Write the bundle; returns what was written and what was missing (a missing input is reported, not invented)."""
    results = Path(cfg["compute"]["runs_csv"]).parent
    out = Path(out_dir) if out_dir else results / "repro"
    out.mkdir(parents=True, exist_ok=True)
    mode = cfg["label_mode"]
    missing: list[str] = []

    lines = sanitize_freeze(pip_freeze() if freeze is None else freeze)
    text = "\n".join(lines) + "\n"
    (out / "pip_freeze.txt").write_text(text, encoding="utf-8")
    lock = root / "requirements-lock.txt"
    if update_lock:
        lock.write_text(text, encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(environment(cfg), indent=2), encoding="utf-8")
    cfgs = copy_configs(out, root)

    from ppfeddata import limitations
    schema_src = next((p for p in limitations.schema_paths(cfg)[:1] if p.exists()), None)
    man_dir = limitations.manifest_dir(cfg)
    if schema_src:
        man_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(schema_src, man_dir / f"feature_schema_{mode}.json")
    elif not (man_dir / f"feature_schema_{mode}.json").exists():
        missing.append(str(limitations.schema_paths(cfg)[0]))

    hashed = [results / "runs.csv", results / "summary.csv", results / "interpretation.json", results / "reports" / "final_report.md", limitations.manifest_path(cfg),
              man_dir / f"feature_schema_{mode}.json", lock, out / "pip_freeze.txt", out / "environment.json"] + [out / c for c in cfgs]
    files = {}
    for p in hashed:
        if p.exists():
            files[_rel(p, root)] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        else:
            missing.append(_rel(p, root))
    manifest = {"label_mode": mode, "git_commit": git_commit(), "files": files, "missing": missing}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"out_dir": str(out), "libraries": len(lines), "configs": len(cfgs), "files_hashed": len(files), "missing": missing, "lock_updated": bool(update_lock)}
