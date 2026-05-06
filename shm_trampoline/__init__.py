"""
memvid SHM Trampoline — Shared-memory DuckDB query engine for memvid datasets.

Exposes an embedded DuckDB query engine over a memory-mapped columnar frame buffer.
External processes can mmap the same file and run SQL queries without full deserialization.

Usage:
    import shm_trampoline as memvid

    # Encode a dataset into a SHM region
    memvid.encode_dataset(data, "/tmp/my_data.shm")

    # Open a DuckDB connection backed by the SHM region
    conn = memvid.open_shm_trampoline("/tmp/my_data.shm")
    result = conn.execute("SELECT * FROM frames WHERE value > 10").fetchall()
"""

from shm_trampoline.core import (
    encode_dataset,
    encode_dataframe,
    open_shm_trampoline,
    ShmRegion,
    ShmTrampolineConnection,
    ShmSchema,
)
from shm_trampoline.schema import (
    infer_schema,
    infer_types,
    map_type_to_duckdb,
    map_type_to_arrow,
    SchemaInferenceError,
)
from shm_trampoline.shm_format import (
    SHM_MAGIC,
    SHM_VERSION,
    SHM_HEADER_SIZE,
    ShmHeader,
    ShmRegionInfo,
)

__all__ = [
    "encode_dataset",
    "encode_dataframe",
    "open_shm_trampoline",
    "ShmRegion",
    "ShmTrampolineConnection",
    "ShmSchema",
    "infer_schema",
    "infer_types",
    "map_type_to_duckdb",
    "map_type_to_arrow",
    "SchemaInferenceError",
    "SHM_MAGIC",
    "SHM_VERSION",
    "SHM_HEADER_SIZE",
    "ShmHeader",
    "ShmRegionInfo",
]
