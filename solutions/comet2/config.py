from dataclasses import dataclass
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent

@dataclass(frozen=True)
class CheckpointConfig:
    path: Path
    model_config: str = 'pi05_b1k'
    asset_id: str = 'behavior-1k/2026-challenge-rgb224'
    revision: str = 'local:group2_20261017_step2498:9afdb712de022c058fc10495f8aca921c24b38291dd40aca9950c38805799923'
    sha256: str = '9afdb712de022c058fc10495f8aca921c24b38291dd40aca9950c38805799923'

@dataclass(frozen=True)
class PolicyConfig:
    robot: str = 'b1k/R1Pro'
    image_size: tuple[int, int] = (224, 224)
    model_horizon: int = 32
    compression_from: int = 26
    execute_steps: int = 20
    action_dim: int = 23
    recovery_rules: Path = HERE / 'assets' / 'recovery_rules.json'

    def __post_init__(self):
        if (self.robot, self.image_size, self.model_horizon, self.compression_from,
                self.execute_steps, self.action_dim) != ('b1k/R1Pro', (224, 224), 32, 26, 20, 23):
            raise ValueError('This version preserves the comet2 inference contract')

MODEL_ENV = {
    'B1K_STAGE_AUX': '1', 'B1K_NORM_STATE': 'q', 'B1K_NORM_ACTIONS': 'z',
    'B1K_QUANTILE_NORM': '0', 'B1K_CORRELATED_NOISE': '0',
    'B1K_NUM_FLOW_SAMPLES': '15', 'B1K_STAGE_LOSS_FIX': '0', 'B1K_FREEZE_VISION': '0',
}

def configure_runtime():

    os.environ.update(MODEL_ENV)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('OPENPI_DATA_HOME', str(HERE / 'runtime' / 'assets'))
    os.environ.setdefault('OPENPI_BEHAVIOR_DATA_ROOT', str(HERE / 'task_metadata'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
