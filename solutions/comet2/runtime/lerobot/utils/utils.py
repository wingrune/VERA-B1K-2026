from __future__ import annotations

import logging
import os
import platform
import select
import subprocess
import sys
import time
from collections.abc import Iterator
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from accelerate import Accelerator


def inside_slurm():


    return "SLURM_JOB_ID" in os.environ


def init_logging(
    log_file: Path | None = None,
    display_pid: bool = False,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    accelerator: Accelerator | None = None,
):


    def custom_format(record: logging.LogRecord) -> str:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        pid_str = f"[PID: {os.getpid()}] " if display_pid else ""
        return f"{record.levelname} {pid_str}{dt} {fnameline[-15:]:>15} {record.getMessage()}"

    formatter = logging.Formatter()
    formatter.format = custom_format

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)


    logger.handlers.clear()


    is_main_process = accelerator.is_main_process if accelerator is not None else True


    if is_main_process:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level.upper())
        logger.addHandler(console_handler)
    else:

        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)


def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def say(text: str, blocking: bool = False):
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]

    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")

    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]

    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    if blocking:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0)


def log_say(text: str, play_sounds: bool = True, blocking: bool = False):
    logging.info(text)

    if play_sounds:
        say(text, blocking)


def get_channel_first_image_shape(image_shape: tuple) -> tuple:
    shape = copy(image_shape)
    if shape[2] < shape[0] and shape[2] < shape[1]:
        shape = (shape[2], shape[0], shape[1])
    elif not (shape[0] < shape[1] and shape[0] < shape[2]):
        raise ValueError(image_shape)

    return shape


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def is_valid_numpy_dtype_string(dtype_str: str) -> bool:

    try:

        np.dtype(dtype_str)
        return True
    except TypeError:

        return False


def enter_pressed() -> bool:
    if platform.system() == "Windows":
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"\r", b"\n")
        return False
    else:
        return select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == ""


def move_cursor_up(lines):

    print(f"\033[{lines}A", end="")


def get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time_s: float):
    days = int(elapsed_time_s // (24 * 3600))
    elapsed_time_s %= 24 * 3600
    hours = int(elapsed_time_s // 3600)
    elapsed_time_s %= 3600
    minutes = int(elapsed_time_s // 60)
    seconds = elapsed_time_s % 60
    return days, hours, minutes, seconds


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:

    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_inner = outdict
        for part in parts[:-1]:
            if part not in d_inner:
                d_inner[part] = {}
            d_inner = d_inner[part]
        d_inner[parts[-1]] = value
    return outdict


def cycle(iterable: Any) -> Iterator[Any]:

    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class SuppressProgressBars:


    def __enter__(self):
        try:
            from datasets.utils.logging import disable_progress_bar

            disable_progress_bar()
        except ImportError:
            logging.getLogger(__name__).debug(
                "SuppressProgressBars is a no-op because 'datasets' is not installed."
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            from datasets.utils.logging import enable_progress_bar

            enable_progress_bar()
        except ImportError:
            pass


class TimerManager:


    def __init__(
        self,
        label: str = "Elapsed-time",
        log: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.label = label
        self.log = log
        self.logger = logger
        self._start: float | None = None
        self._history: list[float] = []

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was never started.")
        elapsed = time.perf_counter() - self._start
        self._history.append(elapsed)
        self._start = None
        if self.log:
            if self.logger is not None:
                self.logger.info(f"{self.label}: {elapsed:.6f} s")
            else:
                logging.info(f"{self.label}: {elapsed:.6f} s")
        return elapsed

    def reset(self):
        self._history.clear()

    @property
    def last(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def avg(self) -> float:
        return mean(self._history) if self._history else 0.0

    @property
    def total(self) -> float:
        return sum(self._history)

    @property
    def count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> list[float]:
        return deepcopy(self._history)

    @property
    def fps_last(self) -> float:
        return 0.0 if self.last == 0 else 1.0 / self.last

    @property
    def fps_avg(self) -> float:
        return 0.0 if self.avg == 0 else 1.0 / self.avg

    def percentile(self, p: float) -> float:

        if not self._history:
            return 0.0
        return float(np.percentile(self._history, p))

    def fps_percentile(self, p: float) -> float:

        val = self.percentile(p)
        return 0.0 if val == 0 else 1.0 / val
