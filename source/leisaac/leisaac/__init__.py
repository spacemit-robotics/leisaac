# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import warnings

# Allow lightweight imports (e.g. tooling, dataset scripts, remote inference clients) to use
# `import leisaac` even when Isaac Sim runtime packages are unavailable on this machine.
try:
	from .utils import monkey_patch
except ModuleNotFoundError as exc:
	if exc.name in {"omni", "isaaclab", "isaaclab_tasks"}:
		warnings.warn(
			f"Isaac Sim runtime not found (`{exc.name}` missing). Skipping IsaacLab monkey patches for this process.",
			RuntimeWarning,
			stacklevel=2,
		)
	else:
		raise

# Register Gym environments only when simulator dependencies are available so non-simulator
# workflows can still import the package without failing during module initialization.
try:
	from .tasks import *
except ModuleNotFoundError as exc:
	if exc.name in {"omni", "isaaclab", "isaaclab_tasks"}:
		warnings.warn(
			f"Isaac Sim runtime not found (`{exc.name}` missing). Skipping task registration for this process.",
			RuntimeWarning,
			stacklevel=2,
		)
	else:
		raise
