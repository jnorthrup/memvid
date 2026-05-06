"""
Binary layout for the SHM trampoline region.

Layout:
    [ SHM Header (SHM_HEADER_SIZE bytes) ]
    [ Arrow IPC stream (variable length)  ]

The SHM header contains a magic number, version, row count, column count,
arrow data offset, and arrow data length. The Arrow IPC stream holds the
columnar data in Apache Arrow IPC format for zero-copy DuckDB integration.

This design allows:
  - Multiple processes to mmap the same file for concurrent reads
  - DuckDB to query the Arrow data directly without deserialization
  - Efficient schema inference from the Arrow schema embedded in IPC
"""

import struct
import json
from dataclasses import dataclass
from typing import Optional

# Binary constants
SHM_MAGIC = b"MVSH"  # Memvid SHM magic number
SHM_VERSION = 1
SHM_HEADER_SIZE = 256  # Fixed header size, padded with zeros


@dataclass
class ShmHeader:
    """Header structure at the beginning of every SHM region.

    Binary layout (all little-endian):
      Offset  Size  Field
      0       4     Magic (b"MVSH")
      4       4     Version (uint32)
      8       8     Row count (uint64)
      16      8     Column count (uint64)
      24      8     Arrow data offset (uint64)
      32      8     Arrow data length (uint64)
      40      8     Schema JSON offset (uint64)
      48      8     Schema JSON length (uint64)
      56      8     Generation counter (uint64)
      64      192   Reserved (zero-padded)
    """
    magic: bytes = SHM_MAGIC
    version: int = SHM_VERSION
    row_count: int = 0
    column_count: int = 0
    arrow_offset: int = 0
    arrow_length: int = 0
    schema_offset: int = 0
    schema_length: int = 0
    generation: int = 0

    # struct format: 4s I 4x Q Q Q Q Q Q Q
    # Size: 4 + 4 + 4(pad) + 8*7 = 68 bytes
    _FORMAT = "<4s I 4x Q Q Q Q Q Q Q"
    _FORMAT_SIZE = struct.calcsize(_FORMAT)  # 68

    def pack(self) -> bytes:
        """Serialize header to bytes (SHM_HEADER_SIZE bytes)."""
        data = struct.pack(
            self._FORMAT,
            self.magic,
            self.version,
            self.row_count,
            self.column_count,
            self.arrow_offset,
            self.arrow_length,
            self.schema_offset,
            self.schema_length,
            self.generation,
        )
        return data.ljust(SHM_HEADER_SIZE, b"\x00")

    @classmethod
    def unpack(cls, data: bytes) -> "ShmHeader":
        """Deserialize header from bytes."""
        if len(data) < SHM_HEADER_SIZE:
            raise ValueError(
                f"SHM header too short: {len(data)} < {SHM_HEADER_SIZE}"
            )
        magic = data[:4]
        if magic != SHM_MAGIC:
            raise ValueError(
                f"Invalid SHM magic: {magic!r} (expected {SHM_MAGIC!r})"
            )
        fields = struct.unpack(cls._FORMAT, data[:cls._FORMAT_SIZE])
        return cls(
            magic=fields[0],
            version=fields[1],
            row_count=fields[2],
            column_count=fields[3],
            arrow_offset=fields[4],
            arrow_length=fields[5],
            schema_offset=fields[6],
            schema_length=fields[7],
            generation=fields[8],
        )


@dataclass
class ShmRegionInfo:
    """Metadata about a SHM region, read from the header."""
    header: ShmHeader
    path: str
    total_size: int = 0
    schema_json: Optional[str] = None

    @property
    def row_count(self) -> int:
        return self.header.row_count

    @property
    def column_count(self) -> int:
        return self.header.column_count

    @property
    def generation(self) -> int:
        return self.header.generation
