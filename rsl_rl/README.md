# Vendored RSL-RL

This directory contains the local training library used by SuperMimic.

- Upstream: https://github.com/leggedrobotics/rsl_rl
- Package: `rsl-rl-lib`, version `5.5.1`
- Source commit: `a87ef38fde4e2a554bec1565e14b868a64de5851`
- License: BSD-3-Clause; retain `LICENSE` and source copyright headers.

## Layout

`rsl_rl/` contains algorithms, runners, models, neural-network modules, rollout
storage, environment interfaces, extensions, and logging utilities. Keep these
modules together: package exports and PPO construction depend on them, including
the distillation, RND, symmetry, CNN, and RNN components.

The nested Git repository and upstream development/release files have been
removed. Runtime Python files are unchanged from the imported snapshot. Track
future modifications in the parent repository.

## Installation

Run `uv sync` from the SuperMimic root. Its `pyproject.toml` selects this directory
as an editable source for `rsl-rl-lib`, so training uses the local code.

To install only this library into an existing compatible environment:

```bash
python -m pip install -e ./rsl_rl
```

TensorBoard support is included; W&B and Neptune remain optional extras.
