# AGENTS.md

Instructions for AI coding agents (Claude Code, Gemini CLI, Codex and others) working in this repository. `CLAUDE.md` and
`GEMINI.md` only import this file, so edit the rules here. People can read it too: it is the short version of how the study is run.
This file is English only on purpose: it is the one exception to the bilingual rule below, so do not create an `AGENTS.vi.md`.

## What the project is

PP-FedData is the code of a study on privacy-preserving federated data generation for an IoT intrusion detection system
(DoS-DDoS-MQTT-IoT dataset). It compares a CVAE trained by FedAvg with DP-SGD (M1), SecAgg+ (M2) or both (M3), and a second
generator, FedDP-Marginal (`MG*` runs), against real-data baselines. Every stage is a sub-command of `python -m ppfeddata.cli`.

Where to read first:

- `README.md` (English) / `README.vi.md` (Vietnamese): overview, results, the commands table ("Running each phase"), run times.
- `docs/USAGE.md` / `docs/USAGE.vi.md`: step-by-step use by goal, the settings people change, own data, troubleshooting. A test
  parses every `python -m ppfeddata.cli` line of both, so a renamed command or option must be changed there too.
- `docs/ARCHITECTURE.md`: components, data flow, code map.
- `docs/PP-FedData_Implementation_Spec.md`: the rules of the study (the spec).
- `docs/SPEC_DEVIATIONS.md`: every deviation from the spec and every decision, with evidence (row codes such as `R.13`).
- The `README.md` of each folder lists its files and how they are produced.

## Environment and commands

```bash
python -m pip install -r requirements.txt   # or requirements-lock.txt for the exact versions of the runs
python -m pip install -e .
python -m ppfeddata.cli --help
python -m pytest -q                          # whole suite, about 15 minutes, needs no raw data
python -m pytest -q tests/test_marginal.py   # one file
```

- Python 3.10 or newer; the runs used Python 3.13 on Windows. If the packages sit in the user site-packages (no virtual
  environment), do not run Python with `-I`: isolated mode hides them and `import numpy` fails.
- Some tests start real Flower simulations with Ray. On Windows a Ray worker can hang (progress stops, CPU idle): stop and rerun.
- `python -m ppfeddata.cli aggregate` takes about 75 s and is deterministic. It rewrites `results/summary.csv`, the figures,
  `results/reports/final_report.md`, `results/interpretation.json` and the generated blocks of both READMEs.
- `python -m ppfeddata.cli accept` checks the Definition of Done against the files (exit 1 if a row fails).
- Experiment runs, searches and the full test suite are long: start them in the background from the beginning, do not move a
  foreground job to the background (that killed the Ray workers once), and do not run other heavy work during timed runs.
- `python -m ppfeddata.cli run --stage trial --dry-run` prints the plan of the experiment matrix without running anything.

## Data: never commit it

- `.githooks/pre-commit` refuses data, run outputs, data or model file types, files over 5 MB and the local working folders.
  Turn it on once per clone with `git config core.hooksPath .githooks`. Never bypass it with `git commit --no-verify` unless a
  person has looked at the refused file and said so. `tests/test_repo_hygiene.py` runs the same hook on every tracked file, so
  `python -m pytest` catches a forbidden file even on a clone where the hook is off. Change the rules in the hook only.
- The raw data (about 12 GB of CSV) are not in the repository and must never be. The path goes in `configs/local.yaml`
  (git-ignored) or in the environment variable `PPFEDDATA_RAW_ROOT`.
- `data/`, `artifacts/`, `data_v1/`, `artifacts_v1/`, `results/runs.csv` and the CSV/JSON intermediates are git-ignored on purpose.
  Never force-add them. The repository is public.
- Never edit the raw data. `tools/extract_normal_pcap.py` is kept byte for byte as it was run (`-text` in `.gitattributes`,
  SHA-256 in `tools/README.md`): do not reformat or touch it.
- The other git-ignored folders in the root (`Latex_FinalCapstone`, `ghi_chu_hoc`) are the team's local working files (the paper
  is its own git repository). Never add them to this repository.

## Generated files: regenerate, do not hand-edit

- The blocks between `<!-- BEGIN GENERATED: ... -->` and `<!-- END GENERATED: ... -->` in `README.md` and `README.vi.md` are
  written by `aggregate` (`readme_gen.py`). Change the generator or the data, then rerun `aggregate`.
- Reports in `results/reports/` and the JSON files in `results/` come from their commands (see the commands table). Fix the
  code that writes them and rerun the command instead of patching the output.

## Documentation rules

- Bilingual pairs mirror each other line by line (same headings, same table rows, same row codes): `README.md` /
  `README.vi.md`, `docs/ARCHITECTURE.md` / `.vi.md`, the spec / `.vi.md`, `docs/SPEC_DEVIATIONS.md` / `.vi.md`. A change to one
  goes into the other in the same edit. Status words map as: ĐÃ DUYỆT = APPROVED, TỰ QUYẾT = OWN CHOICE, ĐÃ GHI = RECORDED,
  ĐÃ SỬA = FIXED, ĐÃ XỬ LÝ = HANDLED, ĐÃ ĐÓNG = CLOSED, ĐÃ QUYẾT = DECIDED, MỞ = OPEN.
- Every change of method, report fix or user decision gets a new row in BOTH deviation logs, same code, same position: at the
  end of the last phase section, or in a new phase section after it. The spec holds rules only, never results.
- Code and reports cite the log by name and row code ("SPEC_DEVIATIONS R.13"), not by path.
- The folder READMEs must list every module, configuration file, test file and committed result; `tests/test_docs.py`
  fails otherwise. Add the line when you add a file.

## Research integrity

- Never invent or estimate a number. Every number in a report or document comes from a file of `results/` or from a run.
  If something blocks a stage, write `BLOCKERS.md` and stop.
- Choices (hyper-parameters, configurations, routes) are made on validation, never on test.
- Report limitations as they are (for example C2ST near 1.0, membership attacks at chance, tuning on non-private data).
  Do not claim that synthetic data improve the IDS, that the attacks show a measured privacy benefit, or that the method is
  novel. A claim about the literature needs sources that were actually read, and must say the search was not systematic.
- When writing about the data, say the Normal PCAP was extracted with the authors' export settings; do not say the group
  "filtered" the data.

## Workflow

- Work one phase or stage at a time and stop at every human gate (G1 `harmonize`, G2 `check`, G3 `baseline`, G4
  `run --stage trial`). `run --stage full --approve-g4` only after a person has read `results/reports/g4_trial.md`.
- Do not rerun expensive stages (searches, round R, the full matrix) unless a person asks for it.
- Do not commit or push unless a person asks. Commit messages carry no AI co-author or "generated with" line.
  For Claude Code, `.claude/settings.json` enforces part of this (it asks before `git push` and before `--no-verify`, and
  turns off the commit attribution); personal settings go in `.claude/settings.local.json` and `CLAUDE.local.md`, which are
  git-ignored.
- After a committed change to code or results, the reproducibility package follows the change: run
  `python -m ppfeddata.cli package --no-update-lock` from a clean tree and commit `results/repro/` on its own. Always pass
  `--no-update-lock`: without it the lock gets the versions installed now, which may differ from those of the runs (idna and
  click already do, SPEC_DEVIATIONS R.14). Drop it only when the libraries were changed on purpose and the study was run
  again with them. If `environment.json` records `git_dirty: true` (the command rewrites `pip_freeze.txt` first when the
  installed libraries changed), commit that and run the command once more.
- The maintainers write in Vietnamese: answer them in Vietnamese unless they ask otherwise. Repository files stay in the
  language of their pair (English `.md`, Vietnamese `.vi.md`).

## Editing pitfalls

- Do not patch files through a shell heredoc that contains backslashes or Vietnamese text: the backslashes collapse (`\b`
  became a backspace, `\n` a real newline) and non-ASCII patterns never match. Use the agent's file edit tool, or write the new
  text to a UTF-8 file first.
- After any scripted patch of a `.py` file, check it with `python -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" <file>`
  and grep it for control characters (`\x07`, `\x08`, `\x0c`).
