"""Zarr v3 codec for unchanged XERO integer restart packets."""
import asyncio
import base64
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import struct
import zlib
import weakref
import numpy as np
from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin
from zarr.registry import register_codec
from . import bridge

NAME = "xero.restart_v1"
MAGIC = b"XEROZ001"
MAX_CHUNK = 512 * 1024**2
STATS = deque(maxlen=64)
_loops = weakref.WeakKeyDictionary()


def _limit():
    # Bound compressed chunks as well as decoded outputs. A Zarr request may
    # otherwise fetch many large restart blobs before the decoder lock is held.
    loop = asyncio.get_running_loop()
    if loop not in _loops:
        _loops[loop] = asyncio.Semaphore(1)
    return _loops[loop]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def unpack_model(encoded, digest):
    if len(encoded) > 32 * 1024**2:
        raise ValueError("Oversized model metadata")
    d = zlib.decompressobj()
    model = d.decompress(base64.b64decode(encoded, validate=True), 16*1024**2+1)
    if len(model) > 16*1024**2 or not d.eof or d.unused_data or sha(model) != digest:
        raise ValueError("Model integrity failure")
    return model


def pack_chunk(packets, origin):
    groups = []
    payload = []
    start = 0
    for g, raw in packets:
        if len(raw) != g["bytes"] or sha(raw) != g["sha256"]:
            raise ValueError("Source packet integrity failure")
        groups.append(dict(start=start, frames=g["frames"], bytes=len(raw), sha256=sha(raw)))
        start += g["frames"]
        payload.append(raw)
    header = canonical(dict(origin=origin, frames=start, groups=groups))
    body = struct.pack("<I", len(header)) + header + b"".join(payload)
    return MAGIC + hashlib.sha256(body).digest() + body, 44 + len(header)


def unpack_header(raw):
    if len(raw) < 44 or len(raw) > MAX_CHUNK or raw[:8] != MAGIC or hashlib.sha256(raw[40:]).digest() != raw[8:40]:
        raise ValueError("Chunk integrity failure")
    n, = struct.unpack_from("<I", raw, 40)
    if not 0 < n <= 1024**2 or n > len(raw)-44:
        raise ValueError("Invalid chunk header")
    return json.loads(raw[44:44+n])


@dataclass(frozen=True)
class XeroRestartCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    model_zlib_base64: str
    model_sha256: str
    profile: str
    restart: int
    height: int
    width: int
    tolerance: float
    is_fixed_size = False

    def __post_init__(self):
        if self.profile not in ("fiarc_integer_experimental_v1", "fiarc_integer_boundscale_v2"):
            raise ValueError("Unsupported numerical profile")
        if type(self.restart) is not int or not 128 <= self.restart <= 1048576 or self.restart & (self.restart-1):
            raise ValueError("Invalid restart interval")
        if not (4 <= self.height <= 4096 and 4 <= self.width <= 4096 and 0 < self.tolerance <= .5):
            raise ValueError("Unsupported geometry/error bound")
        unpack_model(self.model_zlib_base64, self.model_sha256)

    @classmethod
    def from_dict(cls, value):
        if value.get("name") != NAME:
            raise ValueError("Unsupported codec version")
        return cls(**value["configuration"])

    def to_dict(self):
        keys = ("model_zlib_base64", "model_sha256", "profile", "restart", "height", "width", "tolerance")
        return dict(name=NAME, configuration={k: getattr(self, k) for k in keys})

    def validate(self, *, shape, dtype, chunk_grid):
        if len(shape) != 3 or tuple(shape[1:]) != (self.height, self.width) or dtype.to_native_dtype() != np.dtype("float32"):
            raise ValueError("XERO requires float32 [frame,y,x]")
        if chunk_grid.chunk_shape != (min(shape[0], self.restart), self.height, self.width):
            raise ValueError("Chunks must align to restart intervals")

    def compute_encoded_size(self, input_byte_length, chunk_spec):
        raise NotImplementedError("Variable-sized compressed packets")

    async def _encode_single(self, chunk_array, chunk_spec):
        raise NotImplementedError("Repack an existing XERO archive; arbitrary pixel assignment is not supported")

    async def _decode_partial_single(self, byte_getter, selection, chunk_spec):
        async with _limit():
            data = await byte_getter.get(chunk_spec.prototype)
            # Unlike Zarr's usual missing-chunk -> fill behavior, a missing XERO
            # restart is an integrity failure. This applies to ordinary zarr.open.
            if data is None:
                raise FileNotFoundError("Missing XERO restart chunk; refusing to synthesize pixels")
            result = await asyncio.to_thread(self._decode, data.to_bytes(), chunk_spec.shape, selection)
            return chunk_spec.prototype.nd_buffer.from_numpy_array(result)

    async def _decode_single(self, chunk_bytes, chunk_spec):
        result = await asyncio.to_thread(self._decode, chunk_bytes.to_bytes(), chunk_spec.shape,
                                        tuple(slice(0, n) for n in chunk_spec.shape))
        return chunk_spec.prototype.nd_buffer.from_numpy_array(result)

    def _decode(self, raw, shape, selection):
        header = unpack_header(raw)
        count = header["frames"]
        if not 0 < count <= shape[0] or header["origin"] % self.restart:
            raise ValueError("Invalid restart extent")
        slices = []
        squeeze = []
        for axis, (s, n) in enumerate(zip(selection, shape)):
            if isinstance(s, (int, np.integer)):
                s = int(s)
                if s < 0:
                    s += n
                if not 0 <= s < n:
                    raise IndexError(s)
                slices.append((s, s+1, 1)); squeeze.append(axis)
            elif isinstance(s, slice):
                lo, hi, step = s.indices(n)
                if step <= 0:
                    raise ValueError("Use positive-step basic slices")
                slices.append((lo, hi, step))
            else:
                raise IndexError("XERO partial decode supports basic slices")
        out_shape = tuple(len(range(*s)) for s in slices)
        if math.prod(out_shape)*4 > 256*1024**2:
            raise MemoryError("Slice the frame axis: requested chunk output exceeds 256 MiB")
        valid = list(slices)
        valid[0] = (slices[0][0], min(count, slices[0][1]), slices[0][2])
        if any(len(range(*s)) == 0 for s in valid):
            out = np.zeros(out_shape, dtype=np.float32)
        else:
            out, receipt = bridge.decode(self.to_dict()["configuration"], raw, valid)
            STATS.append(dict(origin=header["origin"], valid_frames=count, selected_shape=out_shape, **receipt))
            if out.shape != out_shape:
                padded = np.zeros(out_shape, dtype=np.float32); padded[:len(out)] = out; out = padded
        return np.squeeze(out, axis=tuple(squeeze)) if squeeze else out


register_codec(NAME, XeroRestartCodec)
