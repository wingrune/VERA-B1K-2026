import abc
import builtins
from pathlib import Path
from typing import Any

import draccus

from lerobot.motors.motors_bus import MotorCalibration
from lerobot.types import RobotAction
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from .config import TeleoperatorConfig


class Teleoperator(abc.ABC):


    config_class: builtins.type[TeleoperatorConfig]
    name: str

    def __init__(self, config: TeleoperatorConfig):
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    def __enter__(self):

        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:

        self.disconnect()

    def __del__(self) -> None:

        try:
            if self.is_connected:
                self.disconnect()
        except Exception:
            pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:

        pass

    @property
    @abc.abstractmethod
    def feedback_features(self) -> dict:

        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:

        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:

        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:

        pass

    @abc.abstractmethod
    def calibrate(self) -> None:

        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:

        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:

        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:

        pass

    @abc.abstractmethod
    def get_action(self) -> RobotAction:

        pass

    @abc.abstractmethod
    def send_feedback(self, feedback: dict[str, Any]) -> None:

        pass

    @abc.abstractmethod
    def disconnect(self) -> None:

        pass
