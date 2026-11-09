from dataclasses import dataclass
from typing import Dict, List, Optional


ROBOT_REGISTRY: Dict[str, "RobotConfig"] = {}


def register_robot(name: str, config: "RobotConfig") -> "RobotConfig":

    ROBOT_REGISTRY[name] = config
    return config


@dataclass
class ObservationConfig:

    name: str
    obs_key: str
    dataset_key: str
    resolution: List[int]


@dataclass
class StateActionConfig:

    name: str
    indices: Optional[List[int]] = None
    is_eef: bool = False
    needs_delta_comp: bool = False


@dataclass
class RobotConfig:

    name: str
    robot_type: str
    observations: Dict[str, ObservationConfig]
    action_key: str
    action_dim: int
    action: List[StateActionConfig]
    proprio: List[StateActionConfig]

    def __post_init__(self):


        if self.observations:
            self.observations = {
                key: ObservationConfig(**val) if isinstance(val, dict) else val
                for key, val in self.observations.items()
            }


        self.proprio = [
            StateActionConfig(**item) if isinstance(item, dict) else item
            for item in self.proprio
        ]
        self.action = [
            StateActionConfig(**item) if isinstance(item, dict) else item
            for item in self.action
        ]


        total_dim = sum([len(a.indices) for a in self.action])
        if total_dim != self.action_dim:
            raise ValueError(
                f"Sum of action indices ({total_dim}) doesn't match "
                f"action_dim ({self.action_dim})"
            )
