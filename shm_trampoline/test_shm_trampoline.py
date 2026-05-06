"""
Tests for the SHM trampoline module.

Covers:
  - Schema inference and type mapping
  - SHM region encoding and reading
  - DuckDB query correctness
  - Concurrent reader access
  - Large dataset benchmarks (1M rows, <50ms target)
  - Edge cases (nulls, mixed types, empty data)
"""

import concurrent.futures
import datetime
import json
import math
import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import duckdb
import pyarrow as pa

# Ensure the package is importable
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shm_trampoline import (
    ShmRegion,
    ShmSchema,
    encode_dataset,
    encode_dataframe,
    infer_schema,
    infer_types,
    map_type_to_arrow,
    map_type_to_duckdb,
    open_shm_trampoline,
    SchemaInferenceError,
)
from shm_trampoline.schema import (
    ColumnType,
    ColumnSchema,
    _infer_value_type,
    _promote_types,
    ShmSchema,
)
from shm_trampoline.shm_format import (
    SHM_MAGIC,
    SHM_VERSION,
    SHM_HEADER_SIZE,
    ShmHeader,
    ShmRegionInfo,
)


class TestShmHeader(unittest.TestCase):
    """Test the SHM binary header pack/unpack round-trip."""

    def test_pack_unpack_roundtrip(self):
        h = ShmHeader(
            row_count=1000,
            column_count=5,
            arrow_offset=SHM_HEADER_SIZE,
            arrow_length=4096,
            schema_offset=SHM_HEADER_SIZE + 4096,
            schema_length=256,
            generation=1,
        )
        packed = h.pack()
        self.assertEqual(len(packed), SHM_HEADER_SIZE)
        unpacked = ShmHeader.unpack(packed)
        self.assertEqual(unpacked.magic, SHM_MAGIC)
        self.assertEqual(unpacked.version, SHM_VERSION)
        self.assertEqual(unpacked.row_count, 1000)
        self.assertEqual(unpacked.column_count, 5)
        self.assertEqual(unpacked.arrow_offset, SHM_HEADER_SIZE)
        self.assertEqual(unpacked.arrow_length, 4096)
        self.assertEqual(unpacked.generation, 1)

    def test_invalid_magic(self):
        h = ShmHeader()
        packed = h.pack()
        # Corrupt the magic
        packed = b"XXXX" + packed[4:]
        with self.assertRaises(ValueError) as ctx:
            ShmHeader.unpack(packed)
        self.assertIn("Invalid SHM magic", str(ctx.exception))

    def test_too_short_data(self):
        with self.assertRaises(ValueError):
            ShmHeader.unpack(b"MVSH")


class TestSchemaInference(unittest.TestCase):
    """Test schema inference from raw data."""

    def test_infer_value_type_int(self):
        self.assertEqual(_infer_value_type(0), ColumnType.UINT8)
        self.assertEqual(_infer_value_type(127), ColumnType.UINT8)
        self.assertEqual(_infer_value_type(128), ColumnType.UINT8)  # 128 < 256
        self.assertEqual(_infer_value_type(255), ColumnType.UINT8)  # 255 < 256
        self.assertEqual(_infer_value_type(256), ColumnType.UINT16)  # 256 >= 256
        self.assertEqual(_infer_value_type(-1), ColumnType.INT8)
        self.assertEqual(_infer_value_type(-129), ColumnType.INT16)
        self.assertEqual(_infer_value_type(70000), ColumnType.UINT32)

    def test_infer_value_type_float(self):
        self.assertEqual(_infer_value_type(3.14), ColumnType.FLOAT64)
        self.assertEqual(_infer_value_type(float("nan")), ColumnType.FLOAT64)
        self.assertEqual(_infer_value_type(float("inf")), ColumnType.FLOAT64)

    def test_infer_value_type_other(self):
        self.assertEqual(_infer_value_type(True), ColumnType.BOOL)
        self.assertEqual(_infer_value_type(False), ColumnType.BOOL)
        self.assertEqual(_infer_value_type("hello"), ColumnType.STRING)
        self.assertEqual(_infer_value_type(b"bytes"), ColumnType.BINARY)
        self.assertEqual(_infer_value_type(None), ColumnType.NULL)
        self.assertEqual(
            _infer_value_type(datetime.datetime(2024, 1, 1)), ColumnType.TIMESTAMP
        )
        self.assertEqual(
            _infer_value_type(datetime.date(2024, 1, 1)), ColumnType.DATE
        )

    def test_promote_types_same(self):
        self.assertEqual(_promote_types(ColumnType.INT32, ColumnType.INT32), ColumnType.INT32)

    def test_promote_types_int_widening(self):
        self.assertEqual(_promote_types(ColumnType.INT8, ColumnType.INT16), ColumnType.INT16)
        self.assertEqual(_promote_types(ColumnType.INT16, ColumnType.INT64), ColumnType.INT64)

    def test_promote_types_int_to_float(self):
        self.assertEqual(
            _promote_types(ColumnType.INT32, ColumnType.FLOAT64), ColumnType.FLOAT64
        )

    def test_promote_types_null(self):
        self.assertEqual(
            _promote_types(ColumnType.NULL, ColumnType.INT32), ColumnType.INT32
        )
        self.assertEqual(
            _promote_types(ColumnType.STRING, ColumnType.NULL), ColumnType.STRING
        )

    def test_infer_column(self):
        values = [1, 2, 3, 4, 5]
        schema = infer_types("col", values)
        self.assertEqual(schema.name, "col")
        self.assertIn(schema.col_type, [ColumnType.UINT8, ColumnType.UINT16, ColumnType.INT32])

    def test_infer_column_with_nulls(self):
        values = [1, None, 3, None, 5]
        schema = infer_types("col", values)
        self.assertTrue(schema.nullable)
        self.assertIn(schema.col_type, [ColumnType.UINT8, ColumnType.INT32])

    def test_infer_column_mixed_int_float(self):
        values = [1, 2.5, 3, 4.0]
        schema = infer_types("col", values)
        self.assertEqual(schema.col_type, ColumnType.FLOAT64)

    def test_infer_schema_basic(self):
        data = [
            {"name": "Alice", "age": 30, "score": 95.5},
            {"name": "Bob", "age": 25, "score": 87.3},
            {"name": "Charlie", "age": 35, "score": 92.1},
        ]
        schema = infer_schema(data)
        self.assertEqual(len(schema.columns), 3)
        col_names = [c.name for c in schema.columns]
        self.assertIn("name", col_names)
        self.assertIn("age", col_names)
        self.assertIn("score", col_names)
        name_col = next(c for c in schema.columns if c.name == "name")
        self.assertEqual(name_col.col_type, ColumnType.STRING)
        score_col = next(c for c in schema.columns if c.name == "score")
        self.assertEqual(score_col.col_type, ColumnType.FLOAT64)

    def test_infer_schema_empty(self):
        schema = infer_schema([])
        self.assertEqual(len(schema.columns), 0)

    def test_schema_json_roundtrip(self):
        schema = ShmSchema(
            columns=[
                ColumnSchema("id", ColumnType.INT64, nullable=False),
                ColumnSchema("name", ColumnType.STRING),
            ],
            table_name="test_table",
        )
        json_str = schema.to_json()
        restored = ShmSchema.from_json(json_str)
        self.assertEqual(restored.table_name, "test_table")
        self.assertEqual(len(restored.columns), 2)
        self.assertEqual(restored.columns[0].name, "id")
        self.assertEqual(restored.columns[0].col_type, ColumnType.INT64)
        self.assertFalse(restored.columns[0].nullable)
        self.assertEqual(restored.columns[1].name, "name")
        self.assertEqual(restored.columns[1].col_type, ColumnType.STRING)


class TestTypeMapping(unittest.TestCase):
    """Test type mapping to Arrow and DuckDB types."""

    def test_map_to_arrow(self):
        self.assertEqual(map_type_to_arrow(ColumnType.BOOL), pa.bool_())
        self.assertEqual(map_type_to_arrow(ColumnType.INT8), pa.int8())
        self.assertEqual(map_type_to_arrow(ColumnType.INT16), pa.int16())
        self.assertEqual(map_type_to_arrow(ColumnType.INT32), pa.int32())
        self.assertEqual(map_type_to_arrow(ColumnType.INT64), pa.int64())
        self.assertEqual(map_type_to_arrow(ColumnType.UINT8), pa.uint8())
        self.assertEqual(map_type_to_arrow(ColumnType.UINT16), pa.uint16())
        self.assertEqual(map_type_to_arrow(ColumnType.UINT32), pa.uint32())
        self.assertEqual(map_type_to_arrow(ColumnType.UINT64), pa.uint64())
        self.assertEqual(map_type_to_arrow(ColumnType.FLOAT32), pa.float32())
        self.assertEqual(map_type_to_arrow(ColumnType.FLOAT64), pa.float64())
        self.assertEqual(map_type_to_arrow(ColumnType.STRING), pa.string())

    def test_map_to_duckdb(self):
        self.assertEqual(map_type_to_duckdb(ColumnType.BOOL), "BOOLEAN")
        self.assertEqual(map_type_to_duckdb(ColumnType.INT32), "INTEGER")
        self.assertEqual(map_type_to_duckdb(ColumnType.INT64), "BIGINT")
        self.assertEqual(map_type_to_duckdb(ColumnType.FLOAT64), "DOUBLE")
        self.assertEqual(map_type_to_duckdb(ColumnType.STRING), "VARCHAR")
        self.assertEqual(map_type_to_duckdb(ColumnType.UINT32), "UINTEGER")


class TestEncodeDataset(unittest.TestCase):
    """Test encoding datasets to SHM regions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_encode_basic(self):
        data = [
            {"id": 1, "name": "Alice", "value": 10.5},
            {"id": 2, "name": "Bob", "value": 20.3},
            {"id": 3, "name": "Charlie", "value": 30.1},
        ]
        path = os.path.join(self.tmpdir, "basic.shm")
        info = encode_dataset(data, path)
        self.assertEqual(info.row_count, 3)
        self.assertEqual(info.column_count, 3)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), SHM_HEADER_SIZE)

    def test_encode_empty(self):
        path = os.path.join(self.tmpdir, "empty.shm")
        info = encode_dataset([], path)
        self.assertEqual(info.row_count, 0)

    def test_encode_with_nulls(self):
        data = [
            {"id": 1, "name": "Alice", "value": 10.5},
            {"id": 2, "name": None, "value": None},
            {"id": 3, "name": "Charlie", "value": 30.1},
        ]
        path = os.path.join(self.tmpdir, "nulls.shm")
        info = encode_dataset(data, path)
        self.assertEqual(info.row_count, 3)

    def test_encode_with_schema(self):
        data = [
            {"x": 1, "y": 2},
            {"x": 3, "y": 4},
        ]
        schema = ShmSchema(
            columns=[
                ColumnSchema("x", ColumnType.INT64, nullable=False),
                ColumnSchema("y", ColumnType.INT64, nullable=False),
            ]
        )
        path = os.path.join(self.tmpdir, "schema.shm")
        info = encode_dataset(data, path, schema=schema)
        self.assertEqual(info.row_count, 2)

    def test_encode_creates_parent_dirs(self):
        path = os.path.join(self.tmpdir, "sub", "dir", "data.shm")
        data = [{"x": 1}]
        encode_dataset(data, path)
        self.assertTrue(os.path.exists(path))


class TestShmRegion(unittest.TestCase):
    """Test reading SHM regions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.3},
            {"id": 3, "name": "Charlie", "score": 92.1},
        ]
        self.path = os.path.join(self.tmpdir, "test.shm")
        encode_dataset(self.data, self.path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_open_region(self):
        region = ShmRegion(self.path)
        self.assertEqual(region.row_count, 3)
        self.assertEqual(region.column_count, 3)
        region.close()

    def test_region_context_manager(self):
        with ShmRegion(self.path) as region:
            self.assertEqual(region.row_count, 3)

    def test_region_arrow_table(self):
        with ShmRegion(self.path) as region:
            table = region.arrow_table
            self.assertEqual(len(table), 3)
            self.assertIn("id", table.column_names)
            self.assertIn("name", table.column_names)
            self.assertIn("score", table.column_names)

    def test_region_info(self):
        with ShmRegion(self.path) as region:
            info = region.info()
            self.assertEqual(info.row_count, 3)
            self.assertIsNotNone(info.schema_json)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            ShmRegion("/nonexistent/path.shm")


class TestOpenShmTrampoline(unittest.TestCase):
    """Test the open_shm_trampoline API — the main entry point."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_basic_query(self):
        data = [
            {"id": 1, "name": "Alice", "value": 10},
            {"id": 2, "name": "Bob", "value": 20},
            {"id": 3, "name": "Charlie", "value": 30},
        ]
        path = os.path.join(self.tmpdir, "basic.shm")
        encode_dataset(data, path)

        conn = open_shm_trampoline(path)
        result = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        self.assertEqual(result[0], 3)

        result = conn.execute("SELECT name FROM frames WHERE value > 15").fetchall()
        names = [r[0] for r in result]
        self.assertIn("Bob", names)
        self.assertIn("Charlie", names)
        self.assertNotIn("Alice", names)
        conn.close()

    def test_correct_values(self):
        data = [
            {"x": 100, "y": 200.5},
            {"x": 300, "y": 400.75},
        ]
        path = os.path.join(self.tmpdir, "values.shm")
        encode_dataset(data, path)

        conn = open_shm_trampoline(path)
        row = conn.execute("SELECT x, y FROM frames ORDER BY x").fetchone()
        self.assertEqual(row[0], 100)
        self.assertAlmostEqual(row[1], 200.5)
        conn.close()

    def test_aggregate_query(self):
        data = [{"val": i} for i in range(100)]
        path = os.path.join(self.tmpdir, "agg.shm")
        encode_dataset(data, path)

        conn = open_shm_trampoline(path)
        result = conn.execute("SELECT SUM(val), AVG(val), MIN(val), MAX(val) FROM frames").fetchone()
        self.assertEqual(result[2], 0)  # MIN
        self.assertEqual(result[3], 99)  # MAX
        self.assertEqual(result[0], 4950)  # SUM
        conn.close()

    def test_null_handling(self):
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": None},
            {"id": 3, "name": "Charlie"},
        ]
        path = os.path.join(self.tmpdir, "nulls.shm")
        encode_dataset(data, path)

        conn = open_shm_trampoline(path)
        result = conn.execute("SELECT COUNT(name) FROM frames").fetchone()
        self.assertEqual(result[0], 2)  # COUNT excludes NULL

        result = conn.execute("SELECT id FROM frames WHERE name IS NULL").fetchone()
        self.assertEqual(result[0], 2)
        conn.close()

    def test_custom_table_name(self):
        data = [{"x": 1}]
        path = os.path.join(self.tmpdir, "custom.shm")
        encode_dataset(data, path, table_name="my_table")

        conn = open_shm_trampoline(path)
        result = conn.execute("SELECT COUNT(*) FROM my_table").fetchone()
        self.assertEqual(result[0], 1)
        conn.close()

    def test_where_clause_various_types(self):
        data = [
            {"s": "hello", "i": 42, "f": 3.14, "b": True},
            {"s": "world", "i": 0, "f": 0.0, "b": False},
        ]
        path = os.path.join(self.tmpdir, "types.shm")
        encode_dataset(data, path)

        conn = open_shm_trampoline(path)
        # String comparison
        result = conn.execute("SELECT i FROM frames WHERE s = 'hello'").fetchone()
        self.assertEqual(result[0], 42)
        # Int comparison
        result = conn.execute("SELECT s FROM frames WHERE i > 10").fetchone()
        self.assertEqual(result[0], "hello")
        # Bool comparison
        result = conn.execute("SELECT s FROM frames WHERE b = true").fetchone()
        self.assertEqual(result[0], "hello")
        conn.close()

    def test_return_type_is_duckdb_connection(self):
        from shm_trampoline.core import ShmTrampolineConnection
        data = [{"x": 1}]
        path = os.path.join(self.tmpdir, "rtype.shm")
        encode_dataset(data, path)

        conn = open_shm_trampoline(path)
        self.assertIsInstance(conn, ShmTrampolineConnection)
        # Verify it behaves like a DuckDB connection
        self.assertTrue(hasattr(conn, "execute"))
        self.assertTrue(hasattr(conn, "fetchone"))
        self.assertTrue(hasattr(conn, "fetchall"))
        conn.close()


def _reader_process(path: str, key: str, result_dict: dict) -> None:
    """Module-level function for multiprocessing (must be picklable)."""
    try:
        conn = open_shm_trampoline(path)
        row = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        result_dict[key] = row[0]
        conn.close()
    except Exception as e:
        result_dict[key] = f"ERROR: {e}"


class TestConcurrentReaders(unittest.TestCase):
    """Test that multiple threads can read from the same SHM region concurrently."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data = [{"id": i, "value": i * 10} for i in range(1000)]
        self.path = os.path.join(self.tmpdir, "concurrent.shm")
        encode_dataset(self.data, self.path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_reads(self):
        """Multiple threads open the same SHM file and query concurrently."""
        results = {}
        errors = []

        def reader(thread_id: int):
            try:
                conn = open_shm_trampoline(self.path)
                row = conn.execute("SELECT SUM(value) FROM frames").fetchone()
                results[thread_id] = row[0]
                conn.close()
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(8):
            t = threading.Thread(target=reader, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors in concurrent reads: {errors}")
        expected_sum = sum(row["value"] for row in self.data)
        for tid, val in results.items():
            self.assertEqual(val, expected_sum, f"Thread {tid} got wrong sum")

    def test_concurrent_process_reads(self):
        """Multiple processes can read from the same SHM file."""
        with multiprocessing.Manager() as manager:
            result_dict = manager.dict()
            processes = []
            for i in range(4):
                p = multiprocessing.Process(
                    target=_reader_process,
                    args=(self.path, f"p{i}", result_dict),
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join(timeout=30)

            for i in range(4):
                key = f"p{i}"
                self.assertIn(key, result_dict)
                val = result_dict[key]
                self.assertNotIsInstance(val, str, f"Process {i} errored: {val}")
                self.assertEqual(val, 1000)


class TestBenchmarkLargeDataset(unittest.TestCase):
    """Benchmark: query latency under 50ms for datasets up to 1M rows."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_query_latency_1m_rows(self):
        """Encode 1M rows and verify query latency < 50ms."""
        n = 1_000_000
        # Generate data in chunks to avoid memory spike
        chunk_size = 100_000
        all_data = []
        for i in range(n // chunk_size):
            chunk = [
                {
                    "id": i * chunk_size + j,
                    "category": f"cat_{(i * chunk_size + j) % 100}",
                    "value": float(i * chunk_size + j) * 0.01,
                }
                for j in range(chunk_size)
            ]
            all_data.extend(chunk)

        path = os.path.join(self.tmpdir, "bench_1m.shm")
        print(f"\nEncoding {n:,} rows...")
        t0 = time.perf_counter()
        encode_dataset(all_data, path)
        encode_time = time.perf_counter() - t0
        print(f"Encode time: {encode_time:.2f}s")

        conn = open_shm_trampoline(path)

        # Warm up
        conn.execute("SELECT COUNT(*) FROM frames").fetchone()

        # Benchmark: simple count
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            result = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

        avg_count = sum(times) / len(times)
        print(f"COUNT(*) latency (avg of 5): {avg_count:.2f}ms")
        self.assertLess(avg_count, 50, f"COUNT(*) too slow: {avg_count:.2f}ms")
        self.assertEqual(result[0], n)

        # Benchmark: filtered query
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            result = conn.execute(
                "SELECT SUM(value) FROM frames WHERE category = 'cat_42'"
            ).fetchone()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

        avg_filter = sum(times) / len(times)
        print(f"Filtered SUM latency (avg of 5): {avg_filter:.2f}ms")
        self.assertLess(avg_filter, 50, f"Filtered SUM too slow: {avg_filter:.2f}ms")

        # Benchmark: group by
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            result = conn.execute(
                "SELECT category, COUNT(*), AVG(value) FROM frames GROUP BY category"
            ).fetchall()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

        avg_group = sum(times) / len(times)
        print(f"GROUP BY latency (avg of 5): {avg_group:.2f}ms")
        self.assertLess(avg_group, 50, f"GROUP BY too slow: {avg_group:.2f}ms")

        conn.close()


class TestEncodeDataframe(unittest.TestCase):
    """Test encoding pandas DataFrames."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_encode_pandas_df(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1],
        })
        path = os.path.join(self.tmpdir, "df.shm")
        info = encode_dataframe(df, path)
        self.assertEqual(info.row_count, 3)

        conn = open_shm_trampoline(path)
        result = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        self.assertEqual(result[0], 3)
        conn.close()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_single_row(self):
        data = [{"x": 1, "y": "hello"}]
        path = os.path.join(self.tmpdir, "single.shm")
        encode_dataset(data, path)
        conn = open_shm_trampoline(path)
        result = conn.execute("SELECT * FROM frames").fetchone()
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], "hello")
        conn.close()

    def test_many_columns(self):
        data = [{f"col_{i}": i for i in range(50)}]
        path = os.path.join(self.tmpdir, "wide.shm")
        encode_dataset(data, path)
        conn = open_shm_trampoline(path)
        result = conn.execute("SELECT col_0, col_49 FROM frames").fetchone()
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 49)
        conn.close()

    def test_string_column_with_special_chars(self):
        data = [
            {"text": "Hello, world!"},
            {"text": "Quotes: 'single' \"double\""},
            {"text": "Unicode: \u00e9\u00e8\u00ea\u00eb \u4f60\u597d"},
        ]
        path = os.path.join(self.tmpdir, "strings.shm")
        encode_dataset(data, path)
        conn = open_shm_trampoline(path)
        results = conn.execute("SELECT text FROM frames ORDER BY text").fetchall()
        texts = [r[0] for r in results]
        self.assertEqual(len(texts), 3)
        conn.close()

    def test_all_null_column(self):
        data = [
            {"id": 1, "val": None},
            {"id": 2, "val": None},
        ]
        path = os.path.join(self.tmpdir, "all_null.shm")
        encode_dataset(data, path)
        conn = open_shm_trampoline(path)
        result = conn.execute("SELECT COUNT(*) FROM frames WHERE val IS NULL").fetchone()
        self.assertEqual(result[0], 2)
        conn.close()

    def test_sparse_columns(self):
        """Rows have different sets of columns."""
        data = [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},
            {"b": 5, "c": 6},
        ]
        path = os.path.join(self.tmpdir, "sparse.shm")
        encode_dataset(data, path)
        conn = open_shm_trampoline(path)
        result = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        self.assertEqual(result[0], 3)
        conn.close()


if __name__ == "__main__":
    unittest.main()
