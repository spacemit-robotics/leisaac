from dataclasses import dataclass
from enum import Enum

import torch

# Compatibility with lerobot.async_inference server pickle protocol.
# The remote server unpickles incoming config/observation objects and checks
# against classes imported from lerobot.async_inference.helpers. Expose the
# LeIsaac-side equivalents under the same module path so cross-process pickle
# identity matches without requiring IsaacLab on the inference machine.
_LEROBOT_ASYNC_HELPERS_MODULE = "lerobot.async_inference.helpers"


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple


PolicyFeature.__module__ = _LEROBOT_ASYNC_HELPERS_MODULE


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"


RemotePolicyConfig.__module__ = _LEROBOT_ASYNC_HELPERS_MODULE


RawObservation = dict[str, torch.Tensor]
Action = torch.Tensor


@dataclass
class TimedData:
    """A data object with timestamp and timestep information.

    Args:
        timestamp: Unix timestamp relative to data's creation.
        data: The actual data to wrap a timestamp around.
        timestep: The timestep of the data.
    """

    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False

    def get_observation(self):
        return self.observation


TimedObservation.__module__ = _LEROBOT_ASYNC_HELPERS_MODULE


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self):
        return self.action


TimedAction.__module__ = _LEROBOT_ASYNC_HELPERS_MODULE
