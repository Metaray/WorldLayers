import ctypes
import numpy as np
import os
import sys


__all__ = ['scan_v0_accel', 'scan_v2_accel', 'scan_v13_accel']


if sys.platform == 'win32':
    _lib_name = 'extract_helper.dll'
else:
    _lib_name = 'extract_helper.so'
_accel_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), _lib_name))


_np_bctr_arr = np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C')
_np_bst_arr = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C')
_np_ixm_arr = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C')
_np_u8_arr = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C')


_dsci_v0 = _accel_lib.dsci_v0
_dsci_v0.argtypes = [
    _np_bctr_arr,    # uint64_t *blockCount,
    _np_u8_arr,      # const uint8_t *blocks,
    _np_u8_arr,      # const uint8_t *metadata
]
_dsci_v0.restype = None

def scan_v0_accel(
    blockCount: np.ndarray,
    blocks: np.ndarray,
    metadata: np.ndarray
) -> None:
    _dsci_v0(blockCount, blocks, metadata)


_dsci_v2 = _accel_lib.dsci_v2
_dsci_v2.argtypes = [
    _np_bctr_arr,    # uint64_t *blockCount,
    ctypes.c_uint32, # const uint32_t idLim,
    ctypes.c_uint32, # const uint32_t ySection,
    _np_u8_arr,      # const uint8_t *blocks,
    _np_u8_arr,      # const uint8_t *add,
    _np_u8_arr,      # const uint8_t *add2,
    _np_u8_arr,      # const uint8_t *metadata
]
_dsci_v2.restype = None

def scan_v2_accel(
    blockCount: np.ndarray,
    idLim: int,
    ySection: int,
    blocks: np.ndarray,
    add: np.ndarray,
    add2: np.ndarray,
    metadata: np.ndarray
) -> None:
    _dsci_v2(blockCount, idLim, ySection, blocks, add, add2, metadata)


_dsci_v13 = _accel_lib.dsci_v13
_dsci_v13.argtypes = [
    _np_bctr_arr,    # uint64_t *blockCount,
    ctypes.c_uint32, # const uint32_t idLim,
    ctypes.c_uint32, # const uint32_t ySection,
    _np_bst_arr,     # const uint64_t *blockStates,
    _np_ixm_arr,     # const uint32_t *paletteMap,
    ctypes.c_uint32, # const uint32_t maxPaletteIdx,
    ctypes.c_bool    # const _Bool carry
]
_dsci_v13.restype = None

def scan_v13_accel(
    blockCount: np.ndarray,
    idLim: int,
    ySection: int,
    blockStates: np.ndarray,
    paletteMap: np.ndarray,
    maxPaletteIdx: int,
    carry: bool
) -> None:
    _dsci_v13(blockCount, idLim, ySection, blockStates, paletteMap, maxPaletteIdx, carry)
