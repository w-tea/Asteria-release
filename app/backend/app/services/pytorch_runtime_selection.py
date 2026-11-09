"""Select the portable CUDA wheel without importing an incompatible torch build."""
from __future__ import annotations

from collections.abc import Callable
import ctypes
import sys
from typing import Literal


PyTorchPackageSet = Literal["pytorch-cu126", "pytorch-cu128"]
PYTORCH_PACKAGE_SET_IDS = frozenset({"pytorch-cu126", "pytorch-cu128"})


def detect_cuda_capability() -> tuple[int, int] | None:
    """Query CUDA device 0, respecting CUDA_VISIBLE_DEVICES and CUDA device order.

    Driver initialization/enumeration does not allocate a tensor or GPU context.
    Only the Windows system DLL is loaded, not a DLL from the application folder.
    """
    if sys.platform != "win32":
        return None
    try:
        driver = ctypes.WinDLL("nvcuda.dll", winmode=0x00000800)
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetAttribute.restype = ctypes.c_int
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        if driver.cuInit(0) != 0 or driver.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return None
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR (CUDA Driver API).
        if driver.cuDeviceGetAttribute(ctypes.byref(major), 75, device.value) != 0:
            return None
        if driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, device.value) != 0:
            return None
        return (major.value, minor.value) if major.value > 0 and minor.value >= 0 else None
    except (OSError, AttributeError):
        return None


def select_pytorch_package_set(
    *, capability_probe: Callable[[], tuple[int, int] | None] = detect_cuda_capability,
) -> PyTorchPackageSet:
    capability = capability_probe()
    # PyTorch 2.11 Windows cu128 starts at SM 7.5. The same version's cu126
    # retains Maxwell/Pascal/Volta. Do not downgrade modern or unknown GPUs.
    if capability is not None and (5, 0) <= capability < (7, 5):
        return "pytorch-cu126"
    return "pytorch-cu128"
