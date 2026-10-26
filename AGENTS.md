# Repository Guidelines

## Purpose & Project Structure

This project reproduces SuperMimic using Isaac Lab 3.0 and the Unitree G1 humanoid. Use `docs/2610.28480v1.pdf` as the paper reference and document deviations from its method.

- `src/supermimic/`: installable Python package; `tasks/` contains task discovery, and `assets/` holds robot configurations and USD data.
- `scripts/mimic/`: CSV-to-NPZ conversion, YAML-driven batch conversion, and motion replay.
- `motion_data/`: motion inputs and robot/joint metadata in `config/*.yaml`.
- `rsl_rl/`: vendored editable RSL-RL package; preserve its license and runtime dependency chain.
- `tests/`: pytest tests, currently focused on task registration.
- `skills/` and `.agents/skills/`: contributor and agent workflows.

## Build, Test, and Development Commands

Use Python 3.12 and `uv`. Dependencies resolve to an editable sibling checkout at `../reference/IsaacLab`; update `[tool.uv.sources]` if that location changes.

```bash
uv sync                                     # Install runtime and development dependencies
uv build                                    # Build wheel and source distributions
uv run pytest                               # Run project tests
uv run ruff check src scripts tests          # Lint and check import ordering
uv run ruff format src scripts tests         # Format Python code
uv run isaaclab list_envs --show_presets      # Inspect discovered environments
```

Select optional runtimes with `--extra`, such as `uv run --extra isaacsim ...`. Check task discovery before launching training.

## Coding Style & Naming Conventions

Use four-space indentation, 120-character lines, and Python 3.12 syntax. Follow Ruff's configured lint and import rules in `pyproject.toml`. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Add type hints and concise docstrings to reusable APIs. Resolve project-owned assets through `SUPERMIMIC_ASSETS_DIR`.

## Testing Guidelines

Name tests `tests/test_*.py` and functions `test_*`. Add focused checks for changed behavior; distinguish isolated logic from simulator integration using the configured `unit`, `integration`, `smoke`, and `kitless` markers where applicable. No coverage threshold is configured.

The registration test currently references missing Cartpole task modules, and some asset/motion modules still import external `z_lab` code. Record these blockers when reporting validation results.

## Commit & Pull Request Guidelines

History contains only `Initial commit`, so no established message convention exists. Use concise imperative subjects, such as `Add G1 motion metadata validation`.

Describe the change, link relevant issues, and list validation commands and results. For reproduction changes, record motion datasets, seeds, physics presets, training settings, and relevant metrics or replay evidence. Keep generated checkpoints and large motion outputs out of unrelated changes.
