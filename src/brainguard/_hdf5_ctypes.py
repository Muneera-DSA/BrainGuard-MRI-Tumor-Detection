"""Minimal HDF5 dataset reader via ctypes, used only when h5py is not installed.

Reads whole numeric datasets (any HDF5 storage layout or compression filter the
system libhdf5 supports) into NumPy arrays. Enough for MATLAB v7.3 .mat files.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob

import numpy as np

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    candidates = [ctypes.util.find_library("hdf5"), ctypes.util.find_library("hdf5_serial")]
    candidates += glob.glob("/usr/lib/*/libhdf5_serial.so*") + glob.glob("/usr/lib/*/libhdf5.so*")
    for c in [c for c in candidates if c]:
        try:
            lib = ctypes.CDLL(c)
            break
        except OSError:
            continue
    else:
        raise ImportError("No HDF5 library found: install h5py (pip install h5py)")
    hid = ctypes.c_int64
    lib.H5open.restype = ctypes.c_int
    lib.H5open()
    lib.H5Fopen.restype, lib.H5Fopen.argtypes = hid, [ctypes.c_char_p, ctypes.c_uint, hid]
    lib.H5Dopen2.restype, lib.H5Dopen2.argtypes = hid, [hid, ctypes.c_char_p, hid]
    lib.H5Dget_space.restype, lib.H5Dget_space.argtypes = hid, [hid]
    lib.H5Sget_simple_extent_ndims.restype, lib.H5Sget_simple_extent_ndims.argtypes = ctypes.c_int, [hid]
    lib.H5Sget_simple_extent_dims.restype = ctypes.c_int
    lib.H5Sget_simple_extent_dims.argtypes = [hid, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    lib.H5Dread.restype = ctypes.c_int
    lib.H5Dread.argtypes = [hid, hid, hid, hid, hid, ctypes.c_void_p]
    for f in ("H5Dclose", "H5Sclose", "H5Fclose"):
        getattr(lib, f).argtypes = [hid]
    _lib = lib
    return lib


_NATIVE = {np.dtype(np.float64): "H5T_NATIVE_DOUBLE_g", np.dtype(np.uint16): "H5T_NATIVE_USHORT_g",
           np.dtype(np.int16): "H5T_NATIVE_SHORT_g", np.dtype(np.uint8): "H5T_NATIVE_UCHAR_g"}


def read_datasets(path: str, names_and_dtypes: dict[str, np.dtype]) -> dict[str, np.ndarray]:
    lib = _load()
    H5F_ACC_RDONLY, H5P_DEFAULT, H5S_ALL = 0, 0, 0
    f = lib.H5Fopen(path.encode(), H5F_ACC_RDONLY, H5P_DEFAULT)
    if f < 0:
        raise IOError(f"cannot open {path}")
    out = {}
    try:
        for name, dtype in names_and_dtypes.items():
            dtype = np.dtype(dtype)
            d = lib.H5Dopen2(f, name.encode(), H5P_DEFAULT)
            if d < 0:
                raise KeyError(name)
            sp = lib.H5Dget_space(d)
            nd = lib.H5Sget_simple_extent_ndims(sp)
            dims = (ctypes.c_uint64 * nd)()
            lib.H5Sget_simple_extent_dims(sp, dims, None)
            arr = np.empty(tuple(dims), dtype=dtype)
            mem_type = ctypes.c_int64.in_dll(lib, _NATIVE[dtype]).value
            if lib.H5Dread(d, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr.ctypes.data) < 0:
                raise IOError(f"cannot read {name}")
            lib.H5Sclose(sp)
            lib.H5Dclose(d)
            out[name] = arr
    finally:
        lib.H5Fclose(f)
    return out
