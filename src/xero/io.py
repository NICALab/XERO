"""Explicit single-channel [T,Y,X] or [Z,Y,X] input; no silent reshaping."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np


@contextmanager
def source(path, key='data'):
    path = Path(path)
    suffix = path.suffix.lower()
    handle = None
    try:
        if suffix in ('.h5', '.hdf5'):
            import h5py
            handle = h5py.File(path, 'r')
            array = handle[key]
        elif suffix == '.npy':
            array = np.load(path, mmap_mode='r', allow_pickle=False)
        elif suffix in ('.tif', '.tiff'):
            import tifffile
            try:
                array = tifffile.memmap(path, mode='r')
            except ValueError as exc:
                raise ValueError('TIFF is not memory-mappable. Convert compressed TIFF to an uncompressed stack or HDF5 first.') from exc
        else:
            raise ValueError('Input must be .h5/.hdf5, .npy or uncompressed .tif/.tiff')
        if array.ndim != 3 or min(array.shape) < 1 or min(array.shape[1:]) < 4:
            raise ValueError('Expected one channel with shape [frames,y,x], each spatial side >=4')
        if array.dtype not in (np.dtype('uint16'), np.dtype('uint8'), np.dtype('float32')):
            raise ValueError('Supported source dtypes: uint8, uint16, float32; convert explicitly first')
        yield array
    finally:
        if handle is not None:
            handle.close()


def bound_violations(source, decoded, epsilon):
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(decoded, dtype=np.float64)
    if a.shape != b.shape or not 0 < epsilon <= .5:
        raise ValueError('Invalid shape or bound')
    return int(np.count_nonzero(~np.isfinite(a) | ~np.isfinite(b) | (np.abs(a-b) > epsilon*np.abs(a))))
