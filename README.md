# supermimic

This project reproduces SuperMimic with Isaac Lab 3.0 and Unitree G1. The standalone
model-based learner lives in `src/supermimic_rl/` and does not depend on RSL-RL.

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The registered task is `SuperMimic-MotionTracking-G1`.
See [task configuration and interfaces](docs/motion_tracking_task.md) and the
[reproduction plan](docs/motion_tracking_reproduction_plan.md). See the
[algorithm guide](docs/motion_tracking_algorithm.md) for world-model/actor/critic training and validation.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
./supermimic.sh -i
```

The project pins Python 3.12 in `.python-version` and creates its environment in `.venv`.
Use `uv run <command>` to select it automatically, or activate it with `source .venv/bin/activate`.
The installer runs `uv sync --locked` with the checked-in `uv.lock`. Isaac Lab 3.0.0 is installed
as a published Python package; no Isaac Lab source checkout or fixed directory layout is required.
You can also use the pip-compatible workflow:

```bash
uv venv --python 3.12
uv pip install -e .
```

`uv sync --locked` also installs the development group and reproduces the lockfile.
`uv pip install -e .` installs runtime dependencies from package metadata and resolves their versions
without using `uv.lock`.

The project forwards optional extras from the published Isaac Lab package, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

Place the existing 50 Hz NPZ motions under repository-root `assets/motions/`.
Runtime loads NPZ only; CSV/YAML conversion remains an offline step.

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Standard environment-only CLI, with an optional directory override
uv run isaaclab zero_agent --task SuperMimic-MotionTracking-G1 --num_envs 2 commands.motion.motion_dir=/path/to/npz
```

The tracking task supports the `newton_mjwarp` preset. SuperMimic uses its independent runner, with no PPO prerequisite:

```bash
./supermimic.sh -t --profile smoke --headless
./supermimic.sh -t --profile paper --headless
./supermimic.sh -p --checkpoint /path/to/checkpoint.pt --headless
```

Pass `--motion_dir /path/to/npz` to override the dataset directory. Logs and checkpoints go to `logs/supermimic/`.
The smoke profile preserves the paper networks, k=8 and H=32, using 16 environments and batches of 8.

## Project structure

Task families live in `src/supermimic/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific assets, parameters, and registrations live in `config/g1`. Add another robot by
creating a sibling of `config/g1`; add another task by creating a sibling task-family directory.
`super_motion_tracking/tracking_env_cfg.py` provides shared task composition;
`tracking_env.py` handles reference timing and pre-reset transition boundaries.

## Project assets

Keep project-owned USD files and related data under `src/supermimic/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from supermimic.assets import SUPERMIMIC_ASSETS_DIR

robot_usd_path = SUPERMIMIC_ASSETS_DIR / "robots" / "my_robot.usd"
```

Keep asset configuration in the `supermimic.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
SUPERMIMIC_TEST_MOTION_DIR=/path/to/npz PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_tracking_env.py
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.
The plugin-autoload setting prevents locally installed ROS pytest plugins from interfering with this Python 3.12
environment. Simulator integration tests skip explicitly unless a motion directory is supplied.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
