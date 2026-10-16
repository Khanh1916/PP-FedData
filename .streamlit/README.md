# `.streamlit/` — settings of the Streamlit demo

| File | Role |
|---|---|
| `config.toml` | `gatherUsageStats = false`: Streamlit sends no usage statistics while the demo runs (the project is about privacy) |

Streamlit reads this folder from the **current working directory**. `python -m ppfeddata.cli demo` starts Streamlit in the root of the
repository and also passes `--browser.gatherUsageStats false` on the command line, which wins over any config file or environment
variable; so the demo sends nothing from whatever folder the command is typed in. If you start it yourself with
`streamlit run demo/app.py`, do it from the root of the repository, or this file is not read. The demo itself:
[demo/README.md](../demo/README.md).

The demo needs no secrets, so there is no `secrets.toml`; if one is ever added, it must not be committed.
