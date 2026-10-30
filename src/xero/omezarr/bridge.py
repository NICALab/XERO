"""Bounded persistent decoder pool. Pixels travel over pipes, never via files."""
import atexit
from collections import deque
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import threading
import time

import numpy as np

_lock = threading.RLock()
_session = None


def _jar():
    candidates = [os.environ.get("XERO_DECODER_JAR"),
                  str(Path(__file__).parent / "XERO_ImageJ.jar")]
    for p in candidates:
        if p and Path(p).is_file():
            return str(Path(p).resolve())
    raise RuntimeError("Set XERO_DECODER_JAR to the current XERO_ImageJ.jar (OME-Zarr version).")


def _read(pipe, n):
    out = bytearray(n)
    view = memoryview(out)
    offset = 0
    while offset < n:
        got = pipe.readinto(view[offset:])
        if not got:
            raise EOFError("Decoder stopped before completing the request")
        offset += got
    return out


def _json_read(pipe):
    n, = struct.unpack(">I", _read(pipe, 4))
    if not 0 < n <= 1024 * 1024:
        raise ValueError("Invalid decoder response")
    return json.loads(_read(pipe, n))


def _json_write(pipe, obj):
    data = json.dumps(obj, separators=(",", ":")).encode()
    pipe.write(struct.pack(">I", len(data)))
    pipe.write(data)
    pipe.flush()


class Session:
    def __init__(self, config, backend, jar):
        java = os.environ.get("XERO_JAVA", "java")
        if not shutil.which(java):
            raise RuntimeError("Java 8 or newer is required; set XERO_JAVA to Fiji's java executable if needed.")
        self.errors = deque(maxlen=32)
        self.process = subprocess.Popen([java, "-Xmx2g", "-Dfiarc.cpu.threads=4", "-cp", jar,
            "org.fiarc.strictjava.XeroZarrBridge", backend], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def drain():
            for line in iter(self.process.stderr.readline, b""):
                self.errors.append(line.decode(errors="replace")[-2000:])
        threading.Thread(target=drain, daemon=True).start()
        timer = threading.Timer(float(os.environ.get("XERO_DECODE_TIMEOUT", "900")), self.process.kill)
        timer.daemon = True
        timer.start()
        try:
            _json_write(self.process.stdin, config)
            if not _json_read(self.process.stdout).get("ready"):
                raise RuntimeError("Decoder did not initialize")
        except Exception:
            self.close()
            raise RuntimeError("Decoder initialization failed: " + "".join(self.errors)[-3000:]) from None
        finally:
            timer.cancel()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for p in [self.process.stdin, self.process.stdout, self.process.stderr]:
            p.close()

    def decode(self, raw, selection):
        shape = tuple(len(range(*s)) for s in selection)
        size = int(np.prod(shape, dtype=np.int64)) * 4
        if size > 256 * 1024**2:
            raise MemoryError("Request at most 256 MiB per Zarr chunk; slice the frame axis or use napari.")
        _json_write(self.process.stdin, {"bytes": len(raw), "selection": selection})
        self.process.stdin.write(raw)
        self.process.stdin.flush()
        header = _json_read(self.process.stdout)
        if header.get("bytes") != size:
            raise ValueError("Decoder output shape mismatch")
        data = _read(self.process.stdout, size)
        receipt = _json_read(self.process.stdout)  # Wait for terminal entropy checks.
        return np.frombuffer(data, dtype="<f4").reshape(shape), receipt


def close():
    global _session
    with _lock:
        if _session is not None:
            _session[1].close()
            _session = None


def decode(config, raw, selection):
    global _session
    backend = os.environ.get("XERO_ZARR_BACKEND", "cpu")
    if not re.fullmatch(r"cpu|gpu:[0-9]+|cuda:[0-9]+", backend):
        raise ValueError("XERO_ZARR_BACKEND must be cpu, gpu:N (OpenCL), or cuda:N")
    native = backend.startswith('cuda:')
    jar = None if native else _jar()
    key = (json.dumps(config, sort_keys=True), backend, jar, None if native else Path(jar).stat().st_mtime_ns)
    with _lock:
        started = time.perf_counter()
        # One resident model/engine: switching models releases the previous one.
        reused = _session is not None and _session[0] == key
        timer = threading.Timer(float(os.environ.get("XERO_DECODE_TIMEOUT", "900")),
                                lambda: _session[1].process.kill() if _session else None)
        timer.daemon = True
        try:
            if not reused:
                close()
                if native:
                    from .cuda_bridge import CudaSession
                    _session = (key, CudaSession(config, backend))
                else:
                    _session = (key, Session(config, backend, jar))
            if not native:timer.start()
            result, receipt = _session[1].decode(raw, selection)
            receipt.update(reused_decoder=reused, wall_seconds=time.perf_counter()-started, backend=backend)
            return result, receipt
        except Exception as e:
            detail = "".join(_session[1].errors)[-3000:] if _session else ""
            close()
            raise RuntimeError(f"XERO decode failed: {e}\n{detail}") from e
        finally:
            timer.cancel()


atexit.register(close)
