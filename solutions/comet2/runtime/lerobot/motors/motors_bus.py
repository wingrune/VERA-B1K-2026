from __future__ import annotations

import abc
import logging
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pprint import pformat
from typing import TYPE_CHECKING, Protocol

from tqdm import tqdm

from lerobot.utils.import_utils import _deepdiff_available, _serial_available, require_package

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None

if TYPE_CHECKING or _deepdiff_available:
    from deepdiff import DeepDiff
else:
    DeepDiff = None

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.utils import enter_pressed, move_cursor_up

NameOrID = str | int
Value = int | float

logger = logging.getLogger(__name__)


class MotorsBusBase(abc.ABC):


    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        self.port = port
        self.motors = motors
        self.calibration = calibration if calibration else {}

    @abc.abstractmethod
    def connect(self, handshake: bool = True) -> None:

        pass

    @abc.abstractmethod
    def disconnect(self, disable_torque: bool = True) -> None:

        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:

        pass

    @abc.abstractmethod
    def read(self, data_name: str, motor: str) -> Value:

        pass

    @abc.abstractmethod
    def write(self, data_name: str, motor: str, value: Value) -> None:

        pass

    @abc.abstractmethod
    def sync_read(self, data_name: str, motors: str | list[str] | None = None) -> dict[str, Value]:

        pass

    @abc.abstractmethod
    def sync_write(self, data_name: str, values: dict[str, Value]) -> None:

        pass

    @abc.abstractmethod
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:

        pass

    @abc.abstractmethod
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:

        pass

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]:

        pass

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:

        pass


def get_ctrl_table(model_ctrl_table: dict[str, dict], model: str) -> dict[str, tuple[int, int]]:
    ctrl_table = model_ctrl_table.get(model)
    if ctrl_table is None:
        raise KeyError(f"Control table for {model=} not found.")
    return ctrl_table


def get_address(model_ctrl_table: dict[str, dict], model: str, data_name: str) -> tuple[int, int]:
    ctrl_table = get_ctrl_table(model_ctrl_table, model)
    addr_bytes = ctrl_table.get(data_name)
    if addr_bytes is None:
        raise KeyError(f"Address for '{data_name}' not found in {model} control table.")
    return addr_bytes


def assert_same_address(model_ctrl_table: dict[str, dict], motor_models: list[str], data_name: str) -> None:
    all_addr = []
    all_bytes = []
    for model in motor_models:
        addr, bytes = get_address(model_ctrl_table, model, data_name)
        all_addr.append(addr)
        all_bytes.append(bytes)

    if len(set(all_addr)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different address for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_addr, strict=False))})."
        )

    if len(set(all_bytes)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different bytes representation for `data_name`='{data_name}'"
            f"({list(zip(motor_models, all_bytes, strict=False))})."
        )


class MotorNormMode(str, Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"


@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode
    motor_type_str: str | None = None
    recv_id: int | None = None


class PortHandler(Protocol):
    is_open: bool
    baudrate: int
    packet_start_time: float
    packet_timeout: float
    tx_time_per_byte: float
    is_using: bool
    port_name: str
    ser: serial.Serial

    def __init__(self, port_name: str) -> None: ...

    def openPort(self): ...
    def closePort(self): ...
    def clearPort(self): ...
    def setPortName(self, port_name): ...
    def getPortName(self): ...
    def setBaudRate(self, baudrate): ...
    def getBaudRate(self): ...
    def getBytesAvailable(self): ...
    def readPort(self, length): ...
    def writePort(self, packet): ...
    def setPacketTimeout(self, packet_length): ...
    def setPacketTimeoutMillis(self, msec): ...
    def isPacketTimeout(self): ...
    def getCurrentTime(self): ...
    def getTimeSinceStart(self): ...
    def setupPort(self, cflag_baud): ...
    def getCFlagBaud(self, baudrate): ...


class PacketHandler(Protocol):
    def getTxRxResult(self, result): ...
    def getRxPacketError(self, error): ...
    def txPacket(self, port, txpacket): ...
    def rxPacket(self, port): ...
    def txRxPacket(self, port, txpacket): ...
    def ping(self, port, id): ...
    def action(self, port, id): ...
    def readTx(self, port, id, address, length): ...
    def readRx(self, port, id, length): ...
    def readTxRx(self, port, id, address, length): ...
    def read1ByteTx(self, port, id, address): ...
    def read1ByteRx(self, port, id): ...
    def read1ByteTxRx(self, port, id, address): ...
    def read2ByteTx(self, port, id, address): ...
    def read2ByteRx(self, port, id): ...
    def read2ByteTxRx(self, port, id, address): ...
    def read4ByteTx(self, port, id, address): ...
    def read4ByteRx(self, port, id): ...
    def read4ByteTxRx(self, port, id, address): ...
    def writeTxOnly(self, port, id, address, length, data): ...
    def writeTxRx(self, port, id, address, length, data): ...
    def write1ByteTxOnly(self, port, id, address, data): ...
    def write1ByteTxRx(self, port, id, address, data): ...
    def write2ByteTxOnly(self, port, id, address, data): ...
    def write2ByteTxRx(self, port, id, address, data): ...
    def write4ByteTxOnly(self, port, id, address, data): ...
    def write4ByteTxRx(self, port, id, address, data): ...
    def regWriteTxOnly(self, port, id, address, length, data): ...
    def regWriteTxRx(self, port, id, address, length, data): ...
    def syncReadTx(self, port, start_address, data_length, param, param_length): ...
    def syncWriteTxOnly(self, port, start_address, data_length, param, param_length): ...
    def broadcastPing(self, port): ...


class GroupSyncRead(Protocol):
    port: str
    ph: PortHandler
    start_address: int
    data_length: int
    last_result: bool
    is_param_changed: bool
    param: list
    data_dict: dict

    def __init__(
        self, port: PortHandler, ph: PacketHandler, start_address: int, data_length: int
    ) -> None: ...
    def makeParam(self): ...
    def addParam(self, id): ...
    def removeParam(self, id): ...
    def clearParam(self): ...
    def txPacket(self): ...
    def rxPacket(self): ...
    def txRxPacket(self): ...
    def isAvailable(self, id, address, data_length): ...
    def getData(self, id, address, data_length): ...


class GroupSyncWrite(Protocol):
    port: str
    ph: PortHandler
    start_address: int
    data_length: int
    is_param_changed: bool
    param: list
    data_dict: dict

    def __init__(
        self, port: PortHandler, ph: PacketHandler, start_address: int, data_length: int
    ) -> None: ...
    def makeParam(self): ...
    def addParam(self, id, data): ...
    def removeParam(self, id): ...
    def changeParam(self, id, data): ...
    def clearParam(self): ...
    def txPacket(self): ...


class SerialMotorsBus(MotorsBusBase):


    apply_drive_mode: bool
    available_baudrates: list[int]
    default_baudrate: int
    default_timeout: int
    model_baudrate_table: dict[str, dict]
    model_ctrl_table: dict[str, dict]
    model_encoding_table: dict[str, dict]
    model_number_table: dict[str, int]
    model_resolution_table: dict[str, int]
    normalized_data: list[str]

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        require_package("deepdiff", extra="deepdiff-dep")
        super().__init__(port, motors, calibration)

        self.port_handler: PortHandler
        self.packet_handler: PacketHandler
        self.sync_reader: GroupSyncRead
        self.sync_writer: GroupSyncWrite
        self._comm_success: int
        self._no_error: int

        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict = {m.id: motor for motor, m in self.motors.items()}
        self._model_nb_to_model_dict = {v: k for k, v in self.model_number_table.items()}

        self._validate_motors()

    def __len__(self):
        return len(self.motors)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors: \n{pformat(self.motors, indent=8, sort_dicts=False)},\n"
            ")',\n"
        )

    @cached_property
    def _has_different_ctrl_tables(self) -> bool:
        if len(self.models) < 2:
            return False

        first_table = self.model_ctrl_table[self.models[0]]
        return any(
            DeepDiff(first_table, get_ctrl_table(self.model_ctrl_table, model)) for model in self.models[1:]
        )

    @cached_property
    def models(self) -> list[str]:
        return [m.model for m in self.motors.values()]

    @cached_property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    def _model_nb_to_model(self, motor_nb: int) -> str:
        return self._model_nb_to_model_dict[motor_nb]

    def _id_to_model(self, motor_id: int) -> str:
        return self._id_to_model_dict[motor_id]

    def _id_to_name(self, motor_id: int) -> str:
        return self._id_to_name_dict[motor_id]

    def _get_motor_id(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].id
        elif isinstance(motor, int):
            return motor
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motor_model(self, motor: NameOrID) -> str:
        if isinstance(motor, str):
            return self.motors[motor].model
        elif isinstance(motor, int):
            return self._id_to_model_dict[motor]
        else:
            raise TypeError(f"'{motor}' should be int, str.")

    def _get_motors_list(self, motors: NameOrID | Sequence[NameOrID] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, int):
            return [self._id_to_name(motors)]
        elif isinstance(motors, Sequence):
            return [m if isinstance(m, str) else self._id_to_name(m) for m in motors]
        else:
            raise TypeError(motors)

    def _get_ids_values_dict(self, values: Value | dict[str, Value] | None) -> dict[int, Value]:
        if isinstance(values, (int | float)):
            return dict.fromkeys(self.ids, values)
        elif isinstance(values, dict):
            return {self.motors[motor].id: val for motor, val in values.items()}
        else:
            raise TypeError(f"'values' is expected to be a single value or a dict. Got {values}")

    def _validate_motors(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"Some motors have the same id!\n{self}")


        for model in self.models:
            get_ctrl_table(self.model_ctrl_table, model)

    def _is_comm_success(self, comm: int) -> bool:
        return comm == self._comm_success

    def _is_error(self, error: int) -> bool:
        return error != self._no_error

    def _assert_motors_exist(self) -> None:
        expected_models = {m.id: self.model_number_table[m.model] for m in self.motors.values()}

        found_models = {}
        for id_ in self.ids:
            model_nb = self.ping(id_)
            if model_nb is not None:
                found_models[id_] = model_nb

        missing_ids = [id_ for id_ in self.ids if id_ not in found_models]
        wrong_models = {
            id_: (expected_models[id_], found_models[id_])
            for id_ in found_models
            if expected_models.get(id_) != found_models[id_]
        }

        if missing_ids or wrong_models:
            error_lines = [f"{self.__class__.__name__} motor check failed on port '{self.port}':"]

            if missing_ids:
                error_lines.append("\nMissing motor IDs:")
                error_lines.extend(
                    f"  - {id_} (expected model: {expected_models[id_]})" for id_ in missing_ids
                )

            if wrong_models:
                error_lines.append("\nMotors with incorrect model numbers:")
                error_lines.extend(
                    f"  - {id_} ({self._id_to_name(id_)}): expected {expected}, found {found}"
                    for id_, (expected, found) in wrong_models.items()
                )

            error_lines.append("\nFull expected motor list (id: model_number):")
            error_lines.append(pformat(expected_models, indent=4, sort_dicts=False))
            error_lines.append("\nFull found motor list (id: model_number):")
            error_lines.append(pformat(found_models, indent=4, sort_dicts=False))

            raise RuntimeError("\n".join(error_lines))

    @abc.abstractmethod
    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass

    @property
    def is_connected(self) -> bool:

        return self.port_handler.is_open

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:


        self._connect(handshake)
        self.set_timeout()
        logger.debug(f"{self.__class__.__name__} connected.")

    def _connect(self, handshake: bool = True) -> None:
        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
            elif handshake:
                self._handshake()
        except (FileNotFoundError, OSError, serial.SerialException) as e:
            raise ConnectionError(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from e

    @abc.abstractmethod
    def _handshake(self) -> None:
        pass

    @check_if_not_connected
    def disconnect(self, disable_torque: bool = True) -> None:


        if disable_torque:
            self.port_handler.clearPort()
            self.port_handler.is_using = False
            self.disable_torque(num_retry=5)

        self.port_handler.closePort()
        logger.debug(f"{self.__class__.__name__} disconnected.")

    @classmethod
    def scan_port(cls, port: str, *args, **kwargs) -> dict[int, list[int]]:

        bus = cls(port, {}, *args, **kwargs)
        bus._connect(handshake=False)
        baudrate_ids = {}
        for baudrate in tqdm(bus.available_baudrates, desc="Scanning port"):
            bus.set_baudrate(baudrate)
            ids_models = bus.broadcast_ping()
            if ids_models:
                tqdm.write(f"Motors found for {baudrate=}: {pformat(ids_models, indent=4)}")
                baudrate_ids[baudrate] = list(ids_models)

        bus.port_handler.closePort()
        return baudrate_ids

    def setup_motor(
        self, motor: str, initial_baudrate: int | None = None, initial_id: int | None = None
    ) -> None:

        if not self.is_connected:
            self._connect(handshake=False)

        if initial_baudrate is None:
            initial_baudrate, initial_id = self._find_single_motor(motor)

        if initial_id is None:
            _, initial_id = self._find_single_motor(motor, initial_baudrate)

        model = self.motors[motor].model
        target_id = self.motors[motor].id
        self.set_baudrate(initial_baudrate)
        self._disable_torque(initial_id, model)


        addr, length = get_address(self.model_ctrl_table, model, "ID")
        self._write(addr, length, initial_id, target_id)


        addr, length = get_address(self.model_ctrl_table, model, "Baud_Rate")
        baudrate_value = self.model_baudrate_table[model][self.default_baudrate]
        self._write(addr, length, target_id, baudrate_value)

        self.set_baudrate(self.default_baudrate)

    @abc.abstractmethod
    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        pass

    @abc.abstractmethod
    def configure_motors(self) -> None:

        pass

    @abc.abstractmethod
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:

        pass

    @abc.abstractmethod
    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        pass

    @abc.abstractmethod
    def enable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:

        pass

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):

        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_timeout(self, timeout_ms: int | None = None):

        timeout_ms = timeout_ms if timeout_ms is not None else self.default_timeout
        self.port_handler.setPacketTimeoutMillis(timeout_ms)

    def get_baudrate(self) -> int:

        return self.port_handler.getBaudRate()

    def set_baudrate(self, baudrate: int) -> None:

        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            logger.info(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise RuntimeError("Failed to write bus baud rate.")

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:

        pass

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]:

        pass

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:

        pass

    def reset_calibration(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> None:

        motor_names = self._get_motors_list(motors)

        for motor in motor_names:
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            self.write("Homing_Offset", motor, 0, normalize=False)
            self.write("Min_Position_Limit", motor, 0, normalize=False)
            self.write("Max_Position_Limit", motor, max_res, normalize=False)

        self.calibration = {}

    def set_half_turn_homings(
        self, motors: NameOrID | Sequence[NameOrID] | None = None
    ) -> dict[NameOrID, Value]:

        motor_names = self._get_motors_list(motors)

        self.reset_calibration(motor_names)
        actual_positions = self.sync_read("Present_Position", motor_names, normalize=False)
        homing_offsets = self._get_half_turn_homings(actual_positions)
        for motor, offset in homing_offsets.items():
            self.write("Homing_Offset", motor, offset)

        return homing_offsets

    @abc.abstractmethod
    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        pass

    def record_ranges_of_motion(
        self, motors: NameOrID | Sequence[NameOrID] | None = None, display_values: bool = True
    ) -> tuple[dict[str, Value], dict[str, Value]]:

        motor_names = self._get_motors_list(motors)

        start_positions = self.sync_read("Present_Position", motor_names, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", motor_names, normalize=False)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motor_names:
                    print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:

                move_cursor_up(len(motor_names) + 3)

        same_min_max = [motor for motor in motor_names if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[id_] = -norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[id_] = 100 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                normalized_values[id_] = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError

        return normalized_values

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                unnormalized_values[id_] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError

        return unnormalized_values

    @abc.abstractmethod
    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        pass

    @abc.abstractmethod
    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        pass

    def _serialize_data(self, value: int, length: int) -> list[int]:

        if value < 0:
            raise ValueError(f"Negative values are not allowed: {value}")

        max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(length)
        if max_value is None:
            raise NotImplementedError(f"Unsupported byte size: {length}. Expected [1, 2, 4].")

        if value > max_value:
            raise ValueError(f"Value {value} exceeds the maximum for {length} bytes ({max_value}).")

        return self._split_into_byte_chunks(value, length)

    @abc.abstractmethod
    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:

        pass

    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:

        id_ = self._get_motor_id(motor)
        for n_try in range(1 + num_retry):
            model_number, comm, error = self.packet_handler.ping(self.port_handler, id_)
            if self._is_comm_success(comm):
                break
            logger.debug(f"ping failed for {id_=}: {n_try=} got {comm=} {error=}")

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            else:
                return None
        if self._is_error(error):
            if raise_on_error:
                raise RuntimeError(self.packet_handler.getRxPacketError(error))
            else:
                return None

        return model_number

    @abc.abstractmethod
    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:

        pass

    @check_if_not_connected
    def read(
        self,
        data_name: str,
        motor: str,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> Value:


        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = f"Failed to read '{data_name}' on {id_=} after {num_retry + 1} tries."
        value, _, _ = self._read(addr, length, id_, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

        decoded = self._decode_sign(data_name, {id_: value})

        if normalize and data_name in self.normalized_data:
            normalized = self._normalize(decoded)
            return normalized[id_]

        return decoded[id_]

    def _read(
        self,
        address: int,
        length: int,
        motor_id: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int, int]:
        if length == 1:
            read_fn = self.packet_handler.read1ByteTxRx
        elif length == 2:
            read_fn = self.packet_handler.read2ByteTxRx
        elif length == 4:
            read_fn = self.packet_handler.read4ByteTxRx
        else:
            raise ValueError(length)

        for n_try in range(1 + num_retry):
            value, comm, error = read_fn(self.port_handler, motor_id, address)
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to read @{address=} ({length=}) on {motor_id=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        elif self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")

        return value, comm, error

    @check_if_not_connected
    def write(
        self, data_name: str, motor: str, value: Value, *, normalize: bool = True, num_retry: int = 0
    ) -> None:


        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        int_value = int(value)
        if normalize and data_name in self.normalized_data:
            int_value = self._unnormalize({id_: value})[id_]

        int_value = self._encode_sign(data_name, {id_: int_value})[id_]

        err_msg = f"Failed to write '{data_name}' on {id_=} with '{int_value}' after {num_retry + 1} tries."
        self._write(addr, length, id_, int_value, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

    def _write(
        self,
        addr: int,
        length: int,
        motor_id: int,
        value: int,
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        data = self._serialize_data(value, length)
        for n_try in range(1 + num_retry):
            comm, error = self.packet_handler.writeTxRx(self.port_handler, motor_id, addr, length, data)
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync write @{addr=} ({length=}) on id={motor_id} with {value=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        elif self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")

        return comm, error

    @check_if_not_connected
    def sync_read(
        self,
        data_name: str,
        motors: NameOrID | Sequence[NameOrID] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:


        self._assert_protocol_is_compatible("sync_read")

        names = self._get_motors_list(motors)
        ids = [self.motors[motor].id for motor in names]
        models = [self.motors[motor].model for motor in names]

        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        err_msg = f"Failed to sync read '{data_name}' on {ids=} after {num_retry + 1} tries."
        raw_ids_values, _ = self._sync_read(
            addr, length, ids, num_retry=num_retry, raise_on_error=True, err_msg=err_msg
        )

        decoded = self._decode_sign(data_name, raw_ids_values)

        if normalize and data_name in self.normalized_data:
            normalized = self._normalize(decoded)
            return {self._id_to_name(id_): value for id_, value in normalized.items()}

        return {self._id_to_name(id_): value for id_, value in decoded.items()}

    def _sync_read(
        self,
        addr: int,
        length: int,
        motor_ids: list[int],
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[dict[int, int], int]:
        self._setup_sync_reader(motor_ids, addr, length)
        for n_try in range(1 + num_retry):
            comm = self.sync_reader.txRxPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync read @{addr=} ({length=}) on {motor_ids=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")

        values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
        return values, comm

    def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
        self.sync_reader.clearParam()
        self.sync_reader.start_address = addr
        self.sync_reader.data_length = length
        for id_ in motor_ids:
            self.sync_reader.addParam(id_)


    @check_if_not_connected
    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:


        raw_ids_values = self._get_ids_values_dict(values)
        models = [self._id_to_model(id_) for id_ in raw_ids_values]
        if self._has_different_ctrl_tables:
            assert_same_address(self.model_ctrl_table, models, data_name)

        model = next(iter(models))
        addr, length = get_address(self.model_ctrl_table, model, data_name)

        int_ids_values = {id_: int(val) for id_, val in raw_ids_values.items()}
        if normalize and data_name in self.normalized_data:
            int_ids_values = self._unnormalize(raw_ids_values)

        int_ids_values = self._encode_sign(data_name, int_ids_values)

        err_msg = f"Failed to sync write '{data_name}' with ids_values={int_ids_values} after {num_retry + 1} tries."
        self._sync_write(
            addr, length, int_ids_values, num_retry=num_retry, raise_on_error=True, err_msg=err_msg
        )

    def _sync_write(
        self,
        addr: int,
        length: int,
        ids_values: dict[int, int],
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> int:
        self._setup_sync_writer(ids_values, addr, length)
        for n_try in range(1 + num_retry):
            comm = self.sync_writer.txPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"Failed to sync write @{addr=} ({length=}) with {ids_values=} ({n_try=}): "
                + self.packet_handler.getTxRxResult(comm)
            )

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")

        return comm

    def _setup_sync_writer(self, ids_values: dict[int, int], addr: int, length: int) -> None:
        self.sync_writer.clearParam()
        self.sync_writer.start_address = addr
        self.sync_writer.data_length = length
        for id_, value in ids_values.items():
            data = self._serialize_data(value, length)
            self.sync_writer.addParam(id_, data)


MotorsBus = SerialMotorsBus
