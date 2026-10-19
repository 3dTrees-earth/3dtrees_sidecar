"""Indexed scientific sidecars joined onto a standardized LAZ by point index.

Spatial inference/remapping belongs to the pinned scientific tool. This boundary
never matches coordinates: export records a digest of the method output's points
(header and every standard point field, in order), and assembly joins solely by
point index after the base yields the same digest.
"""

import copy
import hashlib
import json
from pathlib import Path
from uuid import UUID

import laspy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CONTRACT = "3dtrees-scientific-sidecar-proof-v2"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(generation, item):
    UUID(generation)
    if not isinstance(item, int) or item < 1:
        raise ValueError("A positive dataset-item identity is required")
    return {"generation": generation, "item": item}


class _PointsDigest:
    """SHA-256 of a header and every standard point field in point order.

    One running hash per field keeps the digest independent of chunk size."""

    def __init__(self, header):
        crs = header.parse_crs()
        self.header = json.dumps(
            {
                "points": header.point_count,
                "format": header.point_format.id,
                "scales": [float(v) for v in header.scales],
                "offsets": [float(v) for v in header.offsets],
                "crs": crs.to_wkt() if crs else None,
            },
            sort_keys=True,
        )
        self.names = list(header.point_format.standard_dimension_names)
        self.fields = {n: hashlib.sha256() for n in self.names}

    def update(self, chunk):
        for name in self.names:
            self.fields[name].update(np.ascontiguousarray(chunk[name]).tobytes())

    def hexdigest(self):
        total = hashlib.sha256(self.header.encode())
        for name in self.names:
            total.update(name.encode() + self.fields[name].digest())
        return total.hexdigest()


def manifest_of(table):
    """The identity and dimensions a sidecar carries in its Parquet schema."""
    return json.loads((table.schema_arrow.metadata or {})[b"3dtrees"])


def points_sha256(path):
    with laspy.open(path) as reader:
        digest = _PointsDigest(reader.header)
        for chunk in reader.chunk_iterator(CHUNK_POINTS):
            digest.update(chunk)
    return digest.hexdigest()


def unchanged_base(base, enriched):
    if (
        len(base.points) != len(enriched.points)
        or base.header.point_format.id != enriched.header.point_format.id
        or not np.array_equal(base.header.scales, enriched.header.scales)
        or not np.array_equal(base.header.offsets, enriched.header.offsets)
        or base.header.parse_crs() != enriched.header.parse_crs()
    ):
        raise ValueError("Base geometry/header changed")
    for name in base.point_format.dimension_names:
        if name not in enriched.point_format.dimension_names:
            raise ValueError(f"Base dimension missing: {name}")
        before, after = np.asarray(base[name]), np.asarray(enriched[name])
        if before.dtype != after.dtype or before.tobytes() != after.tobytes():
            raise ValueError(f"Base point order or dimension changed: {name}")


# Points per read/write step. LAZ stores 50,000-point chunks and lazrs decodes
# a step's chunks in parallel: one million points ran 2.2x faster than 100,000
# (85 M points, 10 cores) at about 0.5 GB of memory, still independent of size.
CHUNK_POINTS = 1_000_000
# Rows per Parquet row group. Larger groups compress far better than one group
# per point chunk; the buffered rows stay a few tens of MB.
ROW_GROUP_POINTS = 2_000_000


def same_header(a, b):
    if (
        a.point_count != b.point_count
        or a.point_format.id != b.point_format.id
        or not np.array_equal(a.scales, b.scales)
        or not np.array_equal(a.offsets, b.offsets)
        or a.parse_crs() != b.parse_crs()
    ):
        raise ValueError("Base geometry/header changed")


def same_points(a, b, dimensions):
    for name in dimensions:
        before, after = np.asarray(a[name]), np.asarray(b[name])
        if before.dtype != after.dtype or before.tobytes() != after.tobytes():
            raise ValueError(f"Base point order or dimension changed: {name}")


def unchanged_files(base_path, enriched_path):
    with laspy.open(base_path) as a, laspy.open(enriched_path) as b:
        same_header(a.header, b.header)
        dimensions = list(a.header.point_format.dimension_names)
        if not set(dimensions).issubset(b.header.point_format.dimension_names):
            raise ValueError("Base dimension missing")
        for before in a.chunk_iterator(CHUNK_POINTS):
            after = b.read_points(len(before))
            same_points(before, after, dimensions)


class _Rows:
    """Sequential, chunked reads of a sidecar's rows in point order."""

    def __init__(self, path, columns):
        self.batches = pq.ParquetFile(path).iter_batches(
            batch_size=CHUNK_POINTS, columns=["point_index", *columns]
        )
        self.columns, self.pending, self.offset = columns, [], 0

    def take(self, count):
        parts, needed = [], count
        while needed:
            if not self.pending:
                batch = next(self.batches, None)
                if batch is None:
                    raise ValueError("Sidecar rows are fewer than the base points")
                self.pending = [batch]
            batch = self.pending.pop()
            parts.append(batch.slice(0, needed))
            if len(batch) > needed:
                self.pending = [batch.slice(needed)]
            needed -= min(needed, len(batch))
        table = pa.Table.from_batches(parts)
        indices = table["point_index"].to_numpy()
        expected = np.arange(self.offset, self.offset + count, dtype=np.uint64)
        if table["point_index"].null_count or not np.array_equal(indices, expected):
            raise ValueError("Sidecar rows are not in point order")
        self.offset += count
        # Callers that require complete values check this before using them.
        self.missing = [n for n in self.columns if table[n].null_count]
        return {n: table[n].to_numpy() for n in self.columns}

    def finish(self):
        if self.pending or next(self.batches, None) is not None:
            raise ValueError("Sidecar rows are more than the base points")


def export(enriched_path, output, *, generation, item, owned, producer):
    """Store the `owned` dimensions of a method output with each point's index.

    The output is not compared with the base here: its points digest is
    recorded, and assembly refuses the sidecar unless the base matches it."""
    output = Path(output)
    if output.exists():
        raise ValueError("Sidecar destination already exists")
    # A first pass for the digest keeps the manifest in the schema, where it
    # survives any Parquet rewrite of the rows.
    digest = points_sha256(enriched_path)
    with laspy.open(enriched_path) as enriched:
        header = enriched.header
        names = set(header.point_format.dimension_names)
        standard = set(header.point_format.standard_dimension_names)
        if not owned or len(set(owned)) != len(owned) or not set(owned) <= names:
            raise ValueError("Owned dimensions must be named dimensions of the output")
        if set(owned) & standard:
            raise ValueError("Standard point fields cannot be owned")
        dtypes = {
            n: np.dtype(header.point_format.dimension_by_name(n).dtype) for n in owned
        }
        if any(
            n == "point_index" or d.subdtype or d.kind not in "iuf"
            for n, d in dtypes.items()
        ):
            raise ValueError("Only scalar numeric owned dimensions are supported")
        manifest = {
            "contract": CONTRACT,
            **identity(generation, item),
            "points": header.point_count,
            "points_sha256": digest,
            "dimensions": {n: d.str for n, d in dtypes.items()},
            "producer": producer,
        }
        schema = pa.schema(
            [("point_index", pa.uint64())]
            + [(n, pa.from_numpy_dtype(d)) for n, d in dtypes.items()],
            metadata={b"3dtrees": json.dumps(manifest, sort_keys=True).encode()},
        )
        offset = 0
        pending = []
        # The index is 0..n-1 in point order: delta encoding stores it almost
        # for free, while assembly keeps verifying every row's identity.
        with pq.ParquetWriter(
            output,
            schema,
            compression="zstd",
            use_dictionary=list(owned),
            column_encoding={"point_index": "DELTA_BINARY_PACKED"},
        ) as writer:
            for chunk in enriched.chunk_iterator(CHUNK_POINTS):
                columns = {
                    "point_index": np.arange(
                        offset, offset + len(chunk), dtype=np.uint64
                    )
                }
                columns.update({n: np.asarray(chunk[n]) for n in owned})
                pending.append(pa.table(columns, schema=schema))
                offset += len(chunk)
                if sum(len(t) for t in pending) >= ROW_GROUP_POINTS:
                    writer.write_table(
                        pa.concat_tables(pending), row_group_size=ROW_GROUP_POINTS
                    )
                    pending = []
            if pending:
                writer.write_table(
                    pa.concat_tables(pending), row_group_size=ROW_GROUP_POINTS
                )
    return {
        **manifest,
        "sidecar_sha256": sha256(output),
        "bytes": output.stat().st_size,
    }


def _in_point_order(table, count):
    """Whether the row group statistics allow rows 0..count-1 in order, as
    `export` writes them. Statistics cannot prove the order; streaming checks
    every row and falls back to the indexed path when it finds a gap."""
    offset = 0
    column = table.schema_arrow.get_field_index("point_index")
    for number in range(table.metadata.num_row_groups):
        group = table.metadata.row_group(number)
        stats = group.column(column).statistics
        if (
            stats is None
            or not stats.has_min_max
            or stats.min != offset
            or stats.max != offset + group.num_rows - 1
        ):
            return False
        offset += group.num_rows
    return offset == count


class _OutOfOrder(Exception):
    """A sidecar's rows are not exactly the base points in order."""


class _OrderedColumns:
    """A sidecar's owned columns, read in lockstep with the base points."""

    def __init__(self, path, dtypes):
        self.rows = _Rows(path, list(dtypes))
        self.dtypes = dtypes

    def take(self, count):
        try:
            values = self.rows.take(count)
        except ValueError as error:
            raise _OutOfOrder from error
        if self.rows.missing:
            raise ValueError("Scientific dimension dtype mismatch")
        for name, dtype in self.dtypes.items():
            array = values[name]
            if array.dtype != dtype or array.ndim != 1:
                raise ValueError("Scientific dimension dtype mismatch")
        return values

    def finish(self):
        try:
            self.rows.finish()
        except ValueError as error:
            raise _OutOfOrder from error


def assemble(base_path, sidecars, output, *, generation, item):
    """Join sidecars onto their base. Sidecars in point order (all that `export`
    writes) stream in lockstep with the base, in constant memory and without
    scratch; any other row order goes through disk-backed index arrays."""
    output = Path(output)
    if output.exists() or output.resolve() == Path(base_path).resolve():
        raise ValueError("Assembly destination must be new")
    expected = {
        **identity(generation, item),
        "points_sha256": points_sha256(base_path) if sidecars else None,
    }
    with laspy.open(base_path) as base:
        header = copy.deepcopy(base.header)
        count = header.point_count
        if count <= 0:
            raise ValueError("Empty standard base")
        tables = []
        for path in sidecars:
            table = pq.ParquetFile(path)
            manifest = manifest_of(table)
            if manifest.get("contract") != CONTRACT or any(
                manifest.get(k) != expected[k] for k in ("generation", "item")
            ):
                raise ValueError("Sidecar base identity mismatch")
            if manifest.get("points_sha256") != expected["points_sha256"]:
                raise ValueError(
                    "Sidecar points differ from the base: point order, header "
                    "or a base field changed"
                )
            if manifest["points"] != count or table.metadata.num_rows != count:
                raise ValueError("Sidecar point count mismatch")
            if set(table.schema_arrow.names) != {
                "point_index",
                *manifest["dimensions"],
            }:
                raise ValueError("Sidecar column inventory mismatch")
            dtypes = {}
            for name, dtype in manifest["dimensions"].items():
                dtype = np.dtype(dtype)
                if name in header.point_format.dimension_names:
                    raise ValueError("Scientific dimension ownership collision")
                if dtype.subdtype or dtype.kind not in "iuf":
                    raise ValueError("Scientific dimension dtype mismatch")
                header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
                dtypes[name] = dtype
            tables.append((path, table, dtypes))
        streamed = False
        if all(_in_point_order(table, count) for _, table, _ in tables):
            try:
                _assemble_streaming(base, header, tables, output)
                streamed = True
            except _OutOfOrder:
                output.unlink(missing_ok=True)
    if not streamed:
        # The indexed path validates every row and reports what is wrong; it
        # reads the base afresh, since streaming may have consumed part of it.
        with laspy.open(base_path) as base:
            _assemble_indexed(base, header, tables, output, count)
    unchanged_files(base_path, output)
    return {"output_sha256": sha256(output), "points": count}


def _assemble_streaming(base, header, tables, output):
    columns = [_OrderedColumns(path, dtypes) for path, _, dtypes in tables]
    with laspy.open(output, mode="w", header=header) as writer:
        for chunk in base.chunk_iterator(CHUNK_POINTS):
            result = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
            for name in base.header.point_format.dimension_names:
                result[name] = chunk[name]
            for reader in columns:
                for name, values in reader.take(len(chunk)).items():
                    result[name] = values
            writer.write_points(result)
    for reader in columns:
        reader.finish()


def _assemble_indexed(base, header, tables, output, count):
    import tempfile

    with tempfile.TemporaryDirectory(prefix="3dt-point-index-") as directory:
        arrays = {}
        for number, (_, table, dtypes) in enumerate(tables):
            seen = np.memmap(
                Path(directory) / f"{number}-seen",
                mode="w+",
                dtype=np.bool_,
                shape=count,
            )
            owned = {
                # Filenames are internal indices; dimension names never select paths.
                name: np.memmap(
                    Path(directory) / f"{number}-{position}",
                    mode="w+",
                    dtype=dtype,
                    shape=count,
                )
                for position, (name, dtype) in enumerate(dtypes.items())
            }
            for batch in table.iter_batches(batch_size=CHUNK_POINTS):
                indices = batch["point_index"].to_numpy()
                if (
                    batch["point_index"].null_count
                    or indices.dtype != np.dtype("uint64")
                    or np.any(indices >= count)
                    or len(np.unique(indices)) != len(indices)
                    or seen[indices].any()
                ):
                    raise ValueError("Missing, duplicate or invalid point indices")
                seen[indices] = True
                for name, array in owned.items():
                    values = batch[name].to_numpy()
                    if (
                        batch[name].null_count
                        or values.dtype != array.dtype
                        or values.ndim != 1
                    ):
                        raise ValueError("Scientific dimension dtype mismatch")
                    array[indices] = values
            # Row count plus range/uniqueness establishes complete coverage.
            arrays.update(owned)
            del seen
        with laspy.open(output, mode="w", header=header) as writer:
            offset = 0
            for chunk in base.chunk_iterator(CHUNK_POINTS):
                result = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                for name in base.header.point_format.dimension_names:
                    result[name] = chunk[name]
                for name, array in arrays.items():
                    result[name] = array[offset : offset + len(chunk)]
                writer.write_points(result)
                offset += len(chunk)
        arrays.clear()


def merge_for_viewer(paths, output):
    """Convert sequentially, keeping memory proportional to a point chunk."""
    headers = []
    for path in paths:
        with laspy.open(path) as reader:
            headers.append(copy.deepcopy(reader.header))
    header = copy.deepcopy(headers[0])
    if any(
        h.point_format != header.point_format or h.parse_crs() != header.parse_crs()
        for h in headers
    ):
        raise ValueError("Viewer collection requires matching dimensions and CRS")
    header.scales = np.min([h.scales for h in headers], axis=0)
    header.offsets = np.min([h.mins for h in headers], axis=0)
    if Path(output).exists():
        raise ValueError("Viewer destination already exists")
    with laspy.open(output, mode="w", header=header) as writer:
        for path in paths:
            with laspy.open(path) as reader:
                for chunk in reader.chunk_iterator(CHUNK_POINTS):
                    result = laspy.ScaleAwarePointRecord.zeros(
                        len(chunk), header=header
                    )
                    for dimension in chunk.point_format.dimension_names:
                        if dimension not in {"X", "Y", "Z"}:
                            result[dimension] = chunk[dimension]
                    for axis, scale, origin in zip(
                        ("x", "y", "z"), header.scales, header.offsets, strict=True
                    ):
                        values = np.asarray(getattr(chunk, axis))
                        integers = np.rint((values - origin) / scale)
                        if (
                            np.any(integers < -(2**31))
                            or np.any(integers >= 2**31)
                            or not np.allclose(
                                integers * scale + origin, values, rtol=0, atol=1e-8
                            )
                        ):
                            raise ValueError("Viewer merge would change point geometry")
                        result[axis.upper()] = integers.astype(np.int32)
                    writer.write_points(result)
