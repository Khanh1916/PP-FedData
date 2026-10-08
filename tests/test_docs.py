"""The orientation documents (docs/ARCHITECTURE*.md and the README of each folder) name only modules, files and commands that exist, and the
folder READMEs list every module, configuration file and committed result, so they cannot silently fall behind the code."""
import re
import subprocess
from pathlib import Path

import pytest

from ppfeddata.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "ppfeddata"
DOCS = [ROOT / "docs" / "ARCHITECTURE.md", ROOT / "docs" / "ARCHITECTURE.vi.md", PKG / "README.md", ROOT / "configs" / "README.md",
        ROOT / "results" / "README.md", ROOT / "tests" / "README.md", ROOT / "demo" / "README.md"]
NOT_COMMITTED = ("data/", "artifacts/", "configs/local.yaml", "results/runs.csv", "results/summary.csv")
COMMANDS = set(build_parser()._subparsers._group_actions[0].choices)


def _tracked(prefix: str) -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", prefix], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout (e.g. a clean copy without .git): the committed files are unknown")
    return [p for p in out.splitlines() if p]


def _code(md: Path) -> list[str]:
    return re.findall(r"`([^`\n]+)`", md.read_text(encoding="utf-8"))


@pytest.mark.parametrize("md", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_links_resolve(md):
    for target in re.findall(r"\]\(([^)#\s]+)\)", md.read_text(encoding="utf-8")):
        if not target.startswith("http"):
            assert "\\" not in target and (md.parent / target).exists(), f"{md.name}: {target}"


@pytest.mark.parametrize("md", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_named_paths_exist(md):
    for c in _code(md):
        c = c.strip()
        if " " in c or "<" in c or c.startswith(NOT_COMMITTED) or c.endswith(("/*", "/")) and not c.rstrip("/*"):
            continue
        if re.fullmatch(r"(configs|results|src|demo|docs|tests)/[\w./*-]+", c):
            assert list(ROOT.glob(c.rstrip("/"))), f"{md.name}: {c}"
        elif re.fullmatch(r"(data|checks|eval|models|fl)/[\w*]+\.py|[\w*]+\.py", c) and md.parent != ROOT / "tests":
            hits = list(PKG.glob(c)) + list((ROOT / "demo").glob(c)) + list((ROOT / "tests").glob(c))
            assert hits, f"{md.name}: {c}"


def test_package_readme_lists_every_module():
    text = (PKG / "README.md").read_text(encoding="utf-8")
    named = set(re.findall(r"`([\w/]+\.py)`", text))
    for p in sorted(PKG.rglob("*.py")):
        rel = p.relative_to(PKG).as_posix()
        if "__pycache__" in rel or (rel.endswith("__init__.py") and "/" in rel):
            continue
        assert rel in named, f"src/ppfeddata/README.md does not list {rel}"


def test_results_readme_lists_every_committed_result():
    text = (ROOT / "results" / "README.md").read_text(encoding="utf-8")
    for f in _tracked("results"):
        rel = f[len("results/"):]
        if rel.startswith("repro/") or rel == ".gitkeep" or rel == "README.md":
            continue
        name = rel.split("/", 1)[1] if rel.startswith(("reports/", "figures/")) else rel
        assert f"`{name}`" in text, f"results/README.md does not list {rel}"


def test_configs_readme_lists_every_configuration_file():
    text = (ROOT / "configs" / "README.md").read_text(encoding="utf-8")
    for f in _tracked("configs"):
        rel = f[len("configs/"):]
        if rel == "README.md":
            continue
        assert (f"`{rel}`" in text) or (rel.startswith("exp/") and "`exp/*.yaml`" in text), f"configs/README.md does not list {rel}"


@pytest.mark.parametrize("name", ["results/README.md", "configs/README.md"])
def test_the_commands_named_as_producers_exist(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    for row in re.findall(r"^\|[^\n]+\|$", text, flags=re.M):
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) < 3 or set(cells[1]) <= set("-") or cells[1] in ("Command", "Written by"):
            continue
        for cmd in re.findall(r"`([a-z][\w-]*)(?:\s[^`]*)?`", cells[1]):
            assert cmd in COMMANDS, f"{name}: {cmd} is not a CLI command ({row[:80]})"


def test_tests_readme_lists_every_test_file():
    text = (ROOT / "tests" / "README.md").read_text(encoding="utf-8")
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        assert f"`{p.name}`" in text, f"tests/README.md does not list {p.name}"
