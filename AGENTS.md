# Repository Guidelines

## Purpose & Project Structure

This project reproduces SuperMimic using Isaac Lab 3.0 and the Unitree G1 humanoid. Use `docs/2610.28480v1.pdf` as the paper reference and document deviations from its method.

- `src/supermimic/`: installable Python package; `tasks/` contains task discovery, and `assets/` holds robot configurations and USD data.
- `scripts/mimic/`: CSV-to-NPZ conversion, YAML-driven batch conversion, and motion replay.
- `src/supermimic_rl/` and `scripts/supermimic/`: standalone world-model learner and train/play entrypoints.
- `src/supermimic/tasks/super_motion_tracking/rl_cfg/`: task-facing SuperMimic-RL configuration schemas; G1 settings live in `config/g1/agents/supermimic_cfg.py`.
- `motion_data/`: motion inputs and robot/joint metadata in `config/*.yaml`.
- `supermimic_rl/`: standalone learner package; keep algorithm code independent from Isaac Lab's optional RL backends.
- `tests/`: registration, replay/equation contracts, and opt-in Newton integration tests.
- `skills/` and `.agents/skills/`: contributor and agent workflows.

## BeyondMimic Implementation Reference

Use `../reference/whole_body_tracking/` (BeyondMimic's motion-tracking implementation) as a code reference for SuperMimic. This checkout is in the sibling `reference/` directory. Read its `README.md` first.

Within `source/whole_body_tracking/whole_body_tracking/`, consult:

- `tasks/tracking/mdp/`: motion loading, adaptive sampling, tracking commands, observations, rewards, randomization, and terminations.
- `tasks/tracking/tracking_env_cfg.py` and `tasks/tracking/config/g1/flat_env_cfg.py`: environment composition and G1 settings.
- `tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`: reference PPO configuration only.
- `robots/g1.py` and `robots/actuator.py`: robot and actuator settings.

At the reference checkout root, consult `scripts/csv_to_npz.py`, `scripts/replay_npz.py`, and `scripts/rsl_rl/` for motion preprocessing, replay, and training.

The reference targets Isaac Lab 2.1; adapt APIs, quaternion conventions, and backend assumptions to this project's Isaac Lab 3.0 setup. Keep SuperMimic's paper as the method specification and document differences when adapting BeyondMimic code. Preserve upstream license notices when copying code.

## z_lab Isaac Lab 3.0.0 Reference

The sibling `../z_lab/` project contains a BeyondMimic-derived mimic task already migrated to Isaac Lab 3.0.0. Prefer it for current API usage and integration patterns; consult `../reference/whole_body_tracking/` for the original BeyondMimic implementation. SuperMimic's paper remains the method specification.

Within `../z_lab/`, consult:

- `src/z_lab/tasks/mimic/tracking_env_cfg.py` and `src/z_lab/tasks/mimic/mdp/`: migrated environment configuration and tracking terms.
- `src/z_lab/tasks/mimic/config/g1/`: G1 task configuration, registrations (`ZLab-Mimic-Unitree-G1`), and `agents/rsl_rl_ppo_cfg.py`.
- `src/z_lab/assets/robots/g1.py`: robot configuration, joint order, and action scales.
- `scripts/mimic/`: CSV conversion, YAML-driven batch conversion, and NPZ replay.

When adapting code, resolve namespaces and asset paths under `supermimic`, check motion joint ordering and quaternion conventions, and document changes from the BeyondMimic-derived baseline.

## SuperMimic Algorithm Reference

Read `../reference/supermimic_vibed/README.md` before consulting its `supermimic/train.py`, `networks.py`, `buffer.py`, `mdp.py`, and `kinematics.py`. This is an unofficial reproduction with documented deviations; the paper remains authoritative. Preserve this project's BeyondMimic command timing, NPZ-only loading, xyzw order, and CoM velocity contract. Analytic reward/observation/soft-termination heads are intentional. The independent implementation is in `src/supermimic_rl/`; task-registered paper/smoke configs drive `scripts/supermimic/train.py` and `play.py`. Check `docs/motion_tracking_validation.md` for current stage results rather than treating historical audit findings as current behavior.

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

`SuperMimic-MotionTracking-G1` is registered under `super_motion_tracking/`, with shared configuration and G1 specialization in `config/g1/`. Its SuperMimic-RL configuration and standalone train/play scripts are resolved independently of RSL-RL. Runtime reads existing 50 Hz xyzw NPZ files from repository-root `assets/motions/`. See `docs/motion_tracking_task.md` for observation and episode-boundary contracts and `docs/motion_tracking_algorithm.md` for replay, world-model, BPTT, checkpoints, and evaluation. On this machine disable ROS pytest plugin autoload with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## Commit & Pull Request Guidelines

History contains only `Initial commit`, so no established message convention exists. Use concise imperative subjects, such as `Add G1 motion metadata validation`.

Describe the change, link relevant issues, and list validation commands and results. For reproduction changes, record motion datasets, seeds, physics presets, training settings, and relevant metrics or replay evidence. Keep generated checkpoints and large motion outputs out of unrelated changes.
