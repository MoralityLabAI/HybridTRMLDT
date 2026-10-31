"""Read-only GGUF v2/v3 header, metadata and tensor-table reader (stdlib only).

Locates tensor payloads by name for the S0 parity checks. It does not
dequantize and does not load whole tensors: payloads are mmap slices.
Layout follows ggml/docs/gguf.md in PrismML-Eng/llama.cpp at
c96baf4abe24d11a50f326e40b6223b687449d63.
"""
from __future__ import annotations
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
           6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9

# Block size (elements) and bytes per block for the types S0 needs.
BLOCK_LAYOUT = {0: (1, 4), 1: (1, 2), 30: (1, 2), 142: (128, 34), 143: (128, 28)}
TYPE_NAMES = {0: "F32", 1: "F16", 30: "BF16", 142: "PQ2_0", 143: "PTQ1_0"}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]   # ggml order: dims[0] is the contiguous row length
    ggml_type: int
    offset: int             # absolute file offset of the payload

    @property
    def row_length(self) -> int:
        return self.dims[0]

    @property
    def rows(self) -> int:
        n = 1
        for d in self.dims[1:]:
            n *= d
        return n

    @property
    def nbytes(self) -> int | None:
        layout = BLOCK_LAYOUT.get(self.ggml_type)
        if layout is None:
            return None
        block, size = layout
        if self.row_length % block:
            raise ValueError(f"{self.name}: row length {self.row_length} not a multiple of {block}")
        return self.rows * (self.row_length // block) * size


class GGUFFile:
    """mmap-backed reader. Use as a context manager."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._pos = 0
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._fh.close()

    def _take(self, fmt: str):
        size = struct.calcsize(fmt)
        if self._pos + size > len(self._mm):
            raise ValueError("truncated GGUF header")
        (value,) = struct.unpack_from(fmt, self._mm, self._pos)
        self._pos += size
        return value

    def _string(self) -> str:
        n = self._take("<Q")
        if self._pos + n > len(self._mm):
            raise ValueError("truncated GGUF string")
        raw = self._mm[self._pos:self._pos + n]
        self._pos += n
        return raw.decode("utf-8")

    def _value(self, vtype: int):
        if vtype in _SCALAR:
            return self._take(_SCALAR[vtype])
        if vtype == _STRING:
            return self._string()
        if vtype == _ARRAY:
            etype, n = self._take("<I"), self._take("<Q")
            return [self._value(etype) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {vtype}")

    def _parse(self) -> None:
        if self._mm[:4] != MAGIC:
            raise ValueError("not a GGUF file")
        self._pos = 4
        self.version = self._take("<I")
        if self.version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {self.version}")
        n_tensors, n_kv = self._take("<Q"), self._take("<Q")
        self.metadata: dict[str, object] = {}
        for _ in range(n_kv):
            key = self._string()
            self.metadata[key] = self._value(self._take("<I"))
        raw = []
        for _ in range(n_tensors):
            name = self._string()
            n_dims = self._take("<I")
            dims = tuple(self._take("<Q") for _ in range(n_dims))
            raw.append((name, dims, self._take("<I"), self._take("<Q")))
        alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"invalid general.alignment {alignment}")
        self.alignment = alignment
        self.data_start = -(-self._pos // alignment) * alignment
        self.tensors: dict[str, TensorInfo] = {}
        for name, dims, gtype, rel in raw:
            if rel % alignment:
                raise ValueError(f"{name}: offset {rel} not aligned to {alignment}")
            info = TensorInfo(name, dims, gtype, self.data_start + rel)
            size = info.nbytes
            if size is not None and info.offset + size > len(self._mm):
                raise ValueError(f"{name}: payload runs past end of file")
            if name in self.tensors:
                raise ValueError(f"duplicate tensor name {name}")
            self.tensors[name] = info

    def payload(self, name: str, *, first_row: int = 0, rows: int | None = None) -> memoryview:
        """Bytes of whole rows [first_row, first_row + rows) of a known-layout tensor."""
        info = self.tensors[name]
        if info.nbytes is None:
            raise ValueError(f"{name}: type {info.ggml_type} has no known block layout")
        rows = info.rows - first_row if rows is None else rows
        if first_row < 0 or rows <= 0 or first_row + rows > info.rows:
            raise ValueError("row range outside tensor")
        row_bytes = info.nbytes // info.rows
        start = info.offset + first_row * row_bytes
        return memoryview(self._mm)[start:start + rows * row_bytes]
