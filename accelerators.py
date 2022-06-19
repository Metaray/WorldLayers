import ctypes
import numpy as np
import os

with os.add_dll_directory(os.path.dirname(__file__)):
    _accel_dll = ctypes.CDLL('extract_helper.dll')

_np_bctr_arr = np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C')
_np_bst_arr = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C')
_np_ixm_arr = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags='C')
_np_u8_arr = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C')

scan_v0_accel = _accel_dll.dsci_v0
scan_v0_accel.argtypes = [
    _np_bctr_arr,    # uint64_t *blockCount,
    _np_u8_arr,      # const uint8_t *blocks,
    _np_u8_arr,      # const uint8_t *metadata
]
scan_v0_accel.restype = None

scan_v2_accel = _accel_dll.dsci_v2
scan_v2_accel.argtypes = [
    _np_bctr_arr,    # uint64_t *blockCount,
    ctypes.c_uint32, # const uint32_t idLim,
    ctypes.c_uint32, # const uint32_t ySection,
    _np_u8_arr,      # const uint8_t *blocks,
    _np_u8_arr,      # const uint8_t *add,
    _np_u8_arr,      # const uint8_t *add2,
    _np_u8_arr,      # const uint8_t *metadata
]
scan_v2_accel.restype = None

scan_v13_accel = _accel_dll.dsci_v13
scan_v13_accel.argtypes = [
    _np_bctr_arr,    # uint64_t *blockCount,
    ctypes.c_uint32, # const uint32_t idLim,
    ctypes.c_uint32, # const uint32_t ySection,
    _np_bst_arr,     # const uint64_t *blockStates,
    _np_ixm_arr,     # const uint32_t *paletteMap,
    ctypes.c_uint32, # const uint32_t maxPaletteIdx,
    ctypes.c_bool    # const _Bool carry
]
scan_v13_accel.restype = None

__all__ = ['scan_v0_accel', 'scan_v2_accel', 'scan_v13_accel']
