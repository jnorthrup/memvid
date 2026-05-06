"""
Schema inference and type mapping for the SHM trampoline.

Infers column types from raw data, maps them to both Arrow and DuckDB types,
and produces a ShmSchema that describes the structure of the SHM region.
"""

import datetime
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pyarrow as pa


class SchemaInferenceError(Exception):
    """Raised when schema inference fails."""
    pass


class ColumnType(Enum):
    """Supported column types in the SHM trampoline."""
    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BINARY = "binary"
    TIMESTAMP = "timestamp"
    DATE = "date"
    TIME = "time"
    DECIMAL = "decimal"
    NULL = "null"


# Priority-ordered type promotion rules for mixed-type columns
_TYPE_PROMOTION: Dict[Tuple[ColumnType, ColumnType], ColumnType] = {
    # Int promotions
    (ColumnType.INT8, ColumnType.INT16): ColumnType.INT16,
    (ColumnType.INT8, ColumnType.INT32): ColumnType.INT32,
    (ColumnType.INT8, ColumnType.INT64): ColumnType.INT64,
    (ColumnType.INT16, ColumnType.INT32): ColumnType.INT32,
    (ColumnType.INT16, ColumnType.INT64): ColumnType.INT64,
    (ColumnType.INT32, ColumnType.INT64): ColumnType.INT64,
    # UInt promotions
    (ColumnType.UINT8, ColumnType.UINT16): ColumnType.UINT16,
    (ColumnType.UINT8, ColumnType.UINT32): ColumnType.UINT32,
    (ColumnType.UINT8, ColumnType.UINT64): ColumnType.UINT64,
    (ColumnType.UINT16, ColumnType.UINT32): ColumnType.UINT32,
    (ColumnType.UINT16, ColumnType.UINT64): ColumnType.UINT64,
    (ColumnType.UINT32, ColumnType.UINT64): ColumnType.UINT64,
    # Int + UInt -> Float64 (safe)
    (ColumnType.INT8, ColumnType.UINT8): ColumnType.FLOAT64,
    (ColumnType.INT8, ColumnType.UINT16): ColumnType.FLOAT64,
    (ColumnType.INT8, ColumnType.UINT32): ColumnType.FLOAT64,
    (ColumnType.INT8, ColumnType.UINT64): ColumnType.FLOAT64,
    (ColumnType.INT16, ColumnType.UINT8): ColumnType.FLOAT64,
    (ColumnType.INT16, ColumnType.UINT16): ColumnType.FLOAT64,
    (ColumnType.INT16, ColumnType.UINT32): ColumnType.FLOAT64,
    (ColumnType.INT16, ColumnType.UINT64): ColumnType.FLOAT64,
    (ColumnType.INT32, ColumnType.UINT8): ColumnType.FLOAT64,
    (ColumnType.INT32, ColumnType.UINT16): ColumnType.FLOAT64,
    (ColumnType.INT32, ColumnType.UINT32): ColumnType.FLOAT64,
    (ColumnType.INT32, ColumnType.UINT64): ColumnType.FLOAT64,
    (ColumnType.INT64, ColumnType.UINT8): ColumnType.FLOAT64,
    (ColumnType.INT64, ColumnType.UINT16): ColumnType.FLOAT64,
    (ColumnType.INT64, ColumnType.UINT32): ColumnType.FLOAT64,
    (ColumnType.INT64, ColumnType.UINT64): ColumnType.FLOAT64,
    # Int -> Float
    (ColumnType.INT8, ColumnType.FLOAT32): ColumnType.FLOAT64,
    (ColumnType.INT8, ColumnType.FLOAT64): ColumnType.FLOAT64,
    (ColumnType.INT16, ColumnType.FLOAT32): ColumnType.FLOAT64,
    (ColumnType.INT16, ColumnType.FLOAT64): ColumnType.FLOAT64,
    (ColumnType.INT32, ColumnType.FLOAT32): ColumnType.FLOAT64,
    (ColumnType.INT32, ColumnType.FLOAT64): ColumnType.FLOAT64,
    (ColumnType.INT64, ColumnType.FLOAT32): ColumnType.FLOAT64,
    (ColumnType.INT64, ColumnType.FLOAT64): ColumnType.FLOAT64,
    (ColumnType.UINT8, ColumnType.FLOAT32): ColumnType.FLOAT64,
    (ColumnType.UINT8, ColumnType.FLOAT64): ColumnType.FLOAT64,
    (ColumnType.UINT16, ColumnType.FLOAT32): ColumnType.FLOAT64,
    (ColumnType.UINT16, ColumnType.FLOAT64): ColumnType.FLOAT64,
    (ColumnType.UINT32, ColumnType.FLOAT32): ColumnType.FLOAT64,
    (ColumnType.UINT32, ColumnType.FLOAT64): ColumnType.FLOAT64,
    (ColumnType.UINT64, ColumnType.FLOAT32): ColumnType.FLOAT64,
    (ColumnType.UINT64, ColumnType.FLOAT64): ColumnType.FLOAT64,
    # Float promotions
    (ColumnType.FLOAT32, ColumnType.FLOAT64): ColumnType.FLOAT64,
    # Anything with NULL stays the same
    (ColumnType.NULL, ColumnType.BOOL): ColumnType.BOOL,
    (ColumnType.NULL, ColumnType.INT8): ColumnType.INT8,
    (ColumnType.NULL, ColumnType.INT16): ColumnType.INT16,
    (ColumnType.NULL, ColumnType.INT32): ColumnType.INT32,
    (ColumnType.NULL, ColumnType.INT64): ColumnType.INT64,
    (ColumnType.NULL, ColumnType.UINT8): ColumnType.UINT8,
    (ColumnType.NULL, ColumnType.UINT16): ColumnType.UINT16,
    (ColumnType.NULL, ColumnType.UINT32): ColumnType.UINT32,
    (ColumnType.NULL, ColumnType.UINT64): ColumnType.UINT64,
    (ColumnType.NULL, ColumnType.FLOAT32): ColumnType.FLOAT32,
    (ColumnType.NULL, ColumnType.FLOAT64): ColumnType.FLOAT64,
    (ColumnType.NULL, ColumnType.STRING): ColumnType.STRING,
    (ColumnType.NULL, ColumnType.BINARY): ColumnType.BINARY,
    (ColumnType.NULL, ColumnType.TIMESTAMP): ColumnType.TIMESTAMP,
    (ColumnType.NULL, ColumnType.DATE): ColumnType.DATE,
    (ColumnType.NULL, ColumnType.TIME): ColumnType.TIME,
}


@dataclass
class ColumnSchema:
    """Schema for a single column."""
    name: str
    col_type: ColumnType
    nullable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.col_type.value,
            "nullable": self.nullable,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColumnSchema":
        return cls(
            name=d["name"],
            col_type=ColumnType(d["type"]),
            nullable=d.get("nullable", True),
        )


@dataclass
class ShmSchema:
    """Complete schema for a SHM region."""
    columns: List[ColumnSchema] = field(default_factory=list)
    table_name: str = "frames"

    def to_json(self) -> str:
        import json
        return json.dumps({
            "table_name": self.table_name,
            "columns": [c.to_dict() for c in self.columns],
        })

    @classmethod
    def from_json(cls, json_str: str) -> "ShmSchema":
        import json
        data = json.loads(json_str)
        return cls(
            columns=[ColumnSchema.from_dict(c) for c in data["columns"]],
            table_name=data.get("table_name", "frames"),
        )

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]


def _infer_value_type(value: Any) -> ColumnType:
    """Infer the ColumnType of a single Python value."""
    if value is None:
        return ColumnType.NULL
    if isinstance(value, bool):
        return ColumnType.BOOL
    if isinstance(value, int):
        if value < 0:
            if value >= -(2**7):
                return ColumnType.INT8
            if value >= -(2**15):
                return ColumnType.INT16
            if value >= -(2**31):
                return ColumnType.INT32
            return ColumnType.INT64
        else:
            if value < 2**8:
                return ColumnType.UINT8
            if value < 2**16:
                return ColumnType.UINT16
            if value < 2**32:
                return ColumnType.UINT32
            return ColumnType.UINT64
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ColumnType.FLOAT64
        # Use float64 by default for Python floats
        return ColumnType.FLOAT64
    if isinstance(value, str):
        return ColumnType.STRING
    if isinstance(value, bytes):
        return ColumnType.BINARY
    if isinstance(value, datetime.datetime):
        return ColumnType.TIMESTAMP
    if isinstance(value, datetime.date):
        return ColumnType.DATE
    if isinstance(value, datetime.time):
        return ColumnType.TIME
    # Fallback
    return ColumnType.STRING


def _promote_types(a: ColumnType, b: ColumnType) -> ColumnType:
    """Promote two column types to a common type."""
    if a == b:
        return a
    if a == ColumnType.NULL:
        return b
    if b == ColumnType.NULL:
        return a
    # Check both orderings
    key1 = (a, b)
    key2 = (b, a)
    if key1 in _TYPE_PROMOTION:
        return _TYPE_PROMOTION[key1]
    if key2 in _TYPE_PROMOTION:
        return _TYPE_PROMOTION[key2]
    # If no promotion rule exists, fall back to STRING
    return ColumnType.STRING


def infer_types(
    column_name: str,
    values: Sequence[Any],
    sample_size: int = 10000,
) -> ColumnSchema:
    """Infer the type of a column from a sample of values.

    Examines up to `sample_size` values to determine the most specific
    type that can represent all non-null values in the column.
    """
    sample = values[:sample_size] if len(values) > sample_size else values
    current_type = ColumnType.NULL
    has_null = False

    for v in sample:
        if v is None:
            has_null = True
            continue
        val_type = _infer_value_type(v)
        current_type = _promote_types(current_type, val_type)

    if current_type == ColumnType.NULL:
        # All values were null — default to STRING
        current_type = ColumnType.STRING

    return ColumnSchema(
        name=column_name,
        col_type=current_type,
        nullable=has_null or len(values) == 0,
    )


def infer_schema(
    data: Sequence[Dict[str, Any]],
    sample_size: int = 10000,
    table_name: str = "frames",
) -> ShmSchema:
    """Infer a complete schema from a list of row dictionaries.

    Scans up to `sample_size` rows to determine column names and types.
    """
    if not data:
        return ShmSchema(table_name=table_name)

    # Collect all column names (preserving first-seen order)
    seen_names: Dict[str, bool] = {}
    columns: List[str] = []
    for row in data[:sample_size]:
        for key in row:
            if key not in seen_names:
                seen_names[key] = True
                columns.append(key)

    # Infer type for each column
    column_schemas: List[ColumnSchema] = []
    for col_name in columns:
        col_values = [row.get(col_name) for row in data[:sample_size]]
        col_schema = infer_types(col_name, col_values, sample_size)
        column_schemas.append(col_schema)

    return ShmSchema(columns=column_schemas, table_name=table_name)


def map_type_to_arrow(col_type: ColumnType) -> pa.DataType:
    """Map a ColumnType to an Apache Arrow data type."""
    mapping = {
        ColumnType.BOOL: pa.bool_(),
        ColumnType.INT8: pa.int8(),
        ColumnType.INT16: pa.int16(),
        ColumnType.INT32: pa.int32(),
        ColumnType.INT64: pa.int64(),
        ColumnType.UINT8: pa.uint8(),
        ColumnType.UINT16: pa.uint16(),
        ColumnType.UINT32: pa.uint32(),
        ColumnType.UINT64: pa.uint64(),
        ColumnType.FLOAT32: pa.float32(),
        ColumnType.FLOAT64: pa.float64(),
        ColumnType.STRING: pa.string(),
        ColumnType.BINARY: pa.binary(),
        ColumnType.TIMESTAMP: pa.timestamp("us", tz="UTC"),
        ColumnType.DATE: pa.date32(),
        ColumnType.TIME: pa.time64("us"),
        ColumnType.NULL: pa.null(),
    }
    return mapping.get(col_type, pa.string())


def map_type_to_duckdb(col_type: ColumnType) -> str:
    """Map a ColumnType to a DuckDB SQL type string."""
    mapping = {
        ColumnType.BOOL: "BOOLEAN",
        ColumnType.INT8: "TINYINT",
        ColumnType.INT16: "SMALLINT",
        ColumnType.INT32: "INTEGER",
        ColumnType.INT64: "BIGINT",
        ColumnType.UINT8: "UTINYINT",
        ColumnType.UINT16: "USMALLINT",
        ColumnType.UINT32: "UINTEGER",
        ColumnType.UINT64: "UBIGINT",
        ColumnType.FLOAT32: "FLOAT",
        ColumnType.FLOAT64: "DOUBLE",
        ColumnType.STRING: "VARCHAR",
        ColumnType.BINARY: "BLOB",
        ColumnType.TIMESTAMP: "TIMESTAMP",
        ColumnType.DATE: "DATE",
        ColumnType.TIME: "TIME",
        ColumnType.NULL: "VARCHAR",
    }
    return mapping.get(col_type, "VARCHAR")
