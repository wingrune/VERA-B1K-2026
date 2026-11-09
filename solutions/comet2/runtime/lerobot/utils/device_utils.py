import logging

import torch


def auto_select_torch_device() -> torch.device:

    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using mps.")
        return torch.device("mps")
    elif torch.xpu.is_available():
        logging.info("Intel XPU backend detected, using xpu.")
        return torch.device("xpu")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")


def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:

    try_device = str(try_device)
    if try_device.startswith("cuda"):
        assert torch.cuda.is_available()
        device = torch.device(try_device)
    elif try_device == "mps":
        assert torch.backends.mps.is_available()
        device = torch.device("mps")
    elif try_device == "xpu":
        assert torch.xpu.is_available()
        device = torch.device("xpu")
    elif try_device == "cpu":
        device = torch.device("cpu")
        if log:
            logging.warning("Using CPU, this will be slow.")
    else:
        device = torch.device(try_device)
        if log:
            logging.warning(f"Using custom {try_device} device.")
    return device


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):

    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()


            if not device_capability.get("has_fp64", False):
                logging.warning(f"Device {device} does not support float64, using float32 instead.")
                return torch.float32
        else:
            logging.warning(
                f"Device {device} capability check failed. Assuming no support for float64, using float32 instead."
            )
            return torch.float32
        return dtype
    else:
        return dtype


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)
    if try_device.startswith("cuda"):
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "xpu":
        return torch.xpu.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps, xpu or cpu.")


def is_amp_available(device: str):
    if device in ["cuda", "xpu", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")
