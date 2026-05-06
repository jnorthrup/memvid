"""
Core SHM trampoline implementation.

Provides `encode_dataset`, `encode_dataframe`, and `open_shm_trampoline`.
The SHM region is a file-backed memory-mapped region containing an Arrow IPC
stream. DuckDB attaches to it via PyArrow's zero-copy RecordBatchFileReader.

This allows external processes to mmap the same file and run SQL queries
against the data without full deserialization.
"""

import json
import mmap
import os
import struct
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc

from .schema import (
    ColumnType,
    ShmSchema,
    infer_schema,
    infer_types,
    map_type_to_arrow,
)
from .shm_format import SHM_HEADER_SIZE, SHM_MAGIC, SHM_VERSION, ShmHeader, ShmRegionInfo


class ShmRegion:
    """Represents an open SHM region backed by a memory-mapped file.

    Provides access to the Arrow RecordBatch data and metadata.
    Thread-safe for concurrent reads.
    """

    def __init__(self, path: Union[str, Path]):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"SHM region not found: {self._path}")

        self._file = open(self._path, "rb")
        self._mmap: Optional[mmap.mmap] = None
        self._header: Optional[ShmHeader] = None
        self._schema: Optional[ShmSchema] = None
        self._arrow_reader: Optional[ipc.RecordBatchFileReader] = None
        self._lock = threading.Lock()
        self._table: Optional[pa.Table] = None

        self._initialize()

    def _initialize(self) -> None:
        """Read the header and set up the Arrow reader."""
        file_size = os.path.getsize(self._path)
        if file_size < SHM_HEADER_SIZE:
            raise ValueError(
                f"File too small to be a SHM region: {file_size} bytes"
            )

        # Memory-map the file for fast reads
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # Read and validate header
        header_bytes = bytes(self._mmap[:SHM_HEADER_SIZE])
        self._header = ShmHeader.unpack(header_bytes)

        if self._header.schema_offset > 0 and self._header.schema_length > 0:
            schema_data = bytes(
                self._mmap[
                    self._header.schema_offset :
                    self._header.schema_offset + self._header.schema_length
                ]
            )
            self._schema = ShmSchema.from_json(schema_data.decode("utf-8"))

        # Read Arrow data directly from the mmap
        arrow_start = self._header.arrow_offset
        arrow_end = arrow_start + self._header.arrow_length
        arrow_bytes = self._mmap[arrow_start:arrow_end]

        # Create an Arrow reader from the in-memory data
        reader = ipc.RecordBatchStreamReader(pa.BufferReader(arrow_bytes))
        batches = [batch for batch in reader]
        if batches:
            self._table = pa.Table.from_batches(batches)
        else:
            # Empty table with schema
            self._table = pa.Table.from_batches([], schema=reader.schema)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def header(self) -> ShmHeader:
        if self._header is None:
            raise RuntimeError("SHM region not initialized")
        return self._header

    @property
    def schema(self) -> ShmSchema:
        if self._schema is None:
            raise RuntimeError("SHM region has no schema")
        return self._schema

    @property
    def arrow_table(self) -> pa.Table:
        if self._table is None:
            raise RuntimeError("SHM region has no Arrow table")
        return self._table

    @property
    def row_count(self) -> int:
        return self.header.row_count

    @property
    def column_count(self) -> int:
        return self.header.column_count

    def info(self) -> ShmRegionInfo:
        """Get metadata about this SHM region."""
        schema_json = None
        if self._schema is not None:
            schema_json = self._schema.to_json()
        return ShmRegionInfo(
            header=self.header,
            path=str(self._path),
            total_size=os.path.getsize(self._path),
            schema_json=schema_json,
        )

    def close(self) -> None:
        """Release the memory mapping and file handle."""
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                self._close_inner()

    def _close_inner(self) -> None:
        """Inner close — caller holds the lock if one exists."""
        mmap_obj = getattr(self, "_mmap", None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass
            self._mmap = None
        file_obj = getattr(self, "_file", None)
        if file_obj is not None:
            try:
                file_obj.close()
            except Exception:
                pass
            self._file = None
        self._arrow_reader = None
        self._table = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def _build_arrow_table(
    data: Sequence[Dict[str, Any]],
    schema: Optional[ShmSchema] = None,
    sample_size: int = 10000,
) -> pa.Table:
    """Convert a list of row dicts into a PyArrow Table with proper types."""
    if not data:
        return pa.Table.from_pylist([])

    if schema is None:
        schema = infer_schema(data, sample_size=sample_size)

    # Build column-oriented arrays
    columns: Dict[str, list] = {col.name: [] for col in schema.columns}
    for row in data:
        for col in schema.columns:
            columns[col.name].append(row.get(col.name))

    # Build PyArrow arrays with proper types
    arrays = []
    fields = []
    for col in schema.columns:
        arrow_type = map_type_to_arrow(col.col_type)
        values = columns[col.name]
        try:
            arr = pa.array(values, type=arrow_type)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Fall back to string representation
            arr = pa.array([str(v) if v is not None else None for v in values])
        arrays.append(arr)
        fields.append(pa.field(col.name, arr.type))

    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def encode_dataset(
    data: Sequence[Dict[str, Any]],
    path: Union[str, Path],
    schema: Optional[ShmSchema] = None,
    table_name: str = "frames",
    sample_size: int = 10000,
) -> ShmRegionInfo:
    """Encode a dataset (list of dicts) into a SHM region file.

    This creates a memory-mappable file containing:
      - A fixed-size header with metadata
      - A JSON-serialized schema
      - An Arrow IPC stream with the columnar data

    Args:
        data: List of row dictionaries with string keys.
        path: Output file path for the SHM region.
        schema: Optional pre-defined schema. Auto-inferred if None.
        table_name: Logical table name for queries (default "frames").
        sample_size: Number of rows to sample for schema inference.

    Returns:
        ShmRegionInfo with metadata about the created region.
    """
    path = Path(path)

    # Infer or use provided schema
    if schema is None:
        schema = infer_schema(data, sample_size=sample_size, table_name=table_name)
    else:
        schema.table_name = table_name

    # Build Arrow table
    arrow_table = _build_arrow_table(data, schema, sample_size)

    # Serialize Arrow data to IPC stream
    sink = pa.BufferOutputStream()
    writer = ipc.RecordBatchStreamWriter(sink, arrow_table.schema)
    for batch in arrow_table.to_batches():
        writer.write_batch(batch)
    writer.close()
    arrow_bytes = sink.getvalue().to_pybytes()

    # Serialize schema JSON
    schema_json = schema.to_json().encode("utf-8")

    # Build the header
    header = ShmHeader(
        magic=SHM_MAGIC,
        version=SHM_VERSION,
        row_count=len(arrow_table),
        column_count=len(arrow_table.column_names),
        arrow_offset=SHM_HEADER_SIZE,
        arrow_length=len(arrow_bytes),
        schema_offset=SHM_HEADER_SIZE + len(arrow_bytes),
        schema_length=len(schema_json),
        generation=1,
    )

    # Write the file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.pack())
        f.write(arrow_bytes)
        f.write(schema_json)

    return ShmRegionInfo(
        header=header,
        path=str(path),
        total_size=os.path.getsize(path),
        schema_json=schema.to_json(),
    )


def encode_dataframe(
    df: Any,
    path: Union[str, Path],
    table_name: str = "frames",
) -> ShmRegionInfo:
    """Encode a pandas/polars DataFrame into a SHM region file.

    Args:
        df: A pandas DataFrame or any Arrow-compatible object.
        path: Output file path for the SHM region.
        table_name: Logical table name for queries.

    Returns:
        ShmRegionInfo with metadata about the created region.
    """
    # Convert to Arrow table
    if hasattr(df, "to_arrow"):
        arrow_table = df.to_arrow()
    elif hasattr(df, "__dataframe__"):
        # Polars or other DataFrame interchange protocol
        from pyarrow.interchange import from_dataframe
        arrow_table = from_dataframe(df)
    else:
        # pandas or dict-like
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)

    path = Path(path)

    # Build schema from Arrow schema
    columns = []
    for i, field in enumerate(arrow_table.schema):
        col_type = _arrow_type_to_column_type(field.type)
        columns.append(
            {
                "name": field.name,
                "type": col_type.value,
                "nullable": field.nullable,
            }
        )

    schema = ShmSchema(
        columns=[
            __import__(
                "shm_trampoline.schema", fromlist=["ColumnSchema"]
            ).ColumnSchema.from_dict(c)
            for c in columns
        ],
        table_name=table_name,
    )

    # Serialize Arrow data
    sink = pa.BufferOutputStream()
    writer = ipc.RecordBatchStreamWriter(sink, arrow_table.schema)
    for batch in arrow_table.to_batches():
        writer.write_batch(batch)
    writer.close()
    arrow_bytes = sink.getvalue().to_pybytes()

    schema_json = schema.to_json().encode("utf-8")

    header = ShmHeader(
        magic=SHM_MAGIC,
        version=SHM_VERSION,
        row_count=len(arrow_table),
        column_count=len(arrow_table.column_names),
        arrow_offset=SHM_HEADER_SIZE,
        arrow_length=len(arrow_bytes),
        schema_offset=SHM_HEADER_SIZE + len(arrow_bytes),
        schema_length=len(schema_json),
        generation=1,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.pack())
        f.write(arrow_bytes)
        f.write(schema_json)

    return ShmRegionInfo(
        header=header,
        path=str(path),
        total_size=os.path.getsize(path),
        schema_json=schema.to_json(),
    )


def _arrow_type_to_column_type(arrow_type: pa.DataType) -> ColumnType:
    """Map a PyArrow DataType back to a ColumnType."""
    if pa.types.is_boolean(arrow_type):
        return ColumnType.BOOL
    if pa.types.is_int8(arrow_type):
        return ColumnType.INT8
    if pa.types.is_int16(arrow_type):
        return ColumnType.INT16
    if pa.types.is_int32(arrow_type):
        return ColumnType.INT32
    if pa.types.is_int64(arrow_type):
        return ColumnType.INT64
    if pa.types.is_uint8(arrow_type):
        return ColumnType.UINT8
    if pa.types.is_uint16(arrow_type):
        return ColumnType.UINT16
    if pa.types.is_uint32(arrow_type):
        return ColumnType.UINT32
    if pa.types.is_uint64(arrow_type):
        return ColumnType.UINT64
    if pa.types.is_float32(arrow_type):
        return ColumnType.FLOAT32
    if pa.types.is_float64(arrow_type):
        return ColumnType.FLOAT64
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return ColumnType.STRING
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return ColumnType.BINARY
    if pa.types.is_timestamp(arrow_type):
        return ColumnType.TIMESTAMP
    if pa.types.is_date(arrow_type):
        return ColumnType.DATE
    if pa.types.is_time(arrow_type):
        return ColumnType.TIME
    return ColumnType.STRING


def open_shm_trampoline(path: Union[str, Path]) -> "ShmTrampolineConnection":
    """Open a SHM trampoline region and return a DuckDB connection.

    The returned connection has a table named after the schema's table_name
    (default "frames") populated with the data from the SHM region. The data
    is loaded from the memory-mapped Arrow stream without full deserialization.

    Multiple processes can call this concurrently on the same file for
    parallel read access.

    Args:
        path: Path to the SHM region file created by `encode_dataset` or
              `encode_dataframe`.

    Returns:
        A ShmTrampolineConnection (duckdb-compatible) with the SHM data
        available as a queryable table.

    Example:
        >>> conn = open_shm_trampoline("/tmp/data.shm")
        >>> result = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
    """
    region = ShmRegion(path)

    # Create an in-memory DuckDB database and register the Arrow table
    conn = duckdb.connect(":memory:")
    table_name = "frames"
    if region._schema is not None:
        table_name = region._schema.table_name

    # Register the Arrow table — DuckDB reads it zero-copy via Arrow interface
    arrow_table = region.arrow_table
    conn.register(table_name, arrow_table)

    return ShmTrampolineConnection(conn, region)


class ShmTrampolineConnection:
    """Wraps a DuckDB connection with an attached SHM region.

    Delegates all attribute access to the underlying DuckDB connection,
    but also holds a reference to the ShmRegion to keep the mmap alive.
    This object can be used exactly like a duckdb.DuckDBPyConnection.

    The `close()` method closes both the DuckDB connection and the SHM region.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, region: ShmRegion):
        self._conn = conn
        self._region = region

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying DuckDB connection."""
        return getattr(self._conn, name)

    def close(self) -> None:
        """Close both the DuckDB connection and the SHM region."""
        try:
            self._conn.close()
        except Exception:
            pass
        self._region.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def execute(self, *args, **kwargs):
        """Execute a SQL query. Delegates to DuckDB."""
        return self._conn.execute(*args, **kwargs)

    def register(self, *args, **kwargs):
        """Register a Python object as a table. Delegates to DuckDB."""
        return self._conn.register(*args, **kwargs)

    def fetchone(self, *args, **kwargs):
        return self._conn.fetchone(*args, **kwargs)

    def fetchall(self, *args, **kwargs):
        return self._conn.fetchall(*args, **kwargs)

    def fetchdf(self, *args, **kwargs):
        return self._conn.fetchdf(*args, **kwargs)

    def fetch_arrow_table(self, *args, **kwargs):
        return self._conn.fetch_arrow_table(*args, **kwargs)
