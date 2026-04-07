from .base import *

# NOTE:
# service_policy_clients depends on grpc and LeIsaac simulator-side utilities
# (via leisaac.utils.robot_utils -> isaaclab). Importing it unconditionally makes
# lightweight inference-only processes fail on machines that do not host Isaac Sim.
# Keep the client classes available when their dependencies exist, but do not make
# package import itself require the simulator stack.
try:
	from .service_policy_clients import *
except ModuleNotFoundError as exc:
	if exc.name in {"isaaclab", "omni", "isaaclab_tasks"}:
		pass
	else:
		raise
