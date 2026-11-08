from uuid import uuid4

import laspy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyproj import CRS

import sidecar as module_under_test
from sidecar import assemble, export, unchanged_base


@pytest.fixture
def cloud(tmp_path):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(32632))
    header.scales = [0.001, 0.002, 0.003]
    header.offsets = [400000, 5300000, 200]
    base = laspy.LasData(header)
    base.X = [0, 0, 2, 3, 4, 5]
    base.Y = [0, 0, 1, 1, 2, 2]
    base.Z = [0, 0, 3, 4, 5, 6]
    base.intensity = [7, 8, 9, 10, 11, 12]
    base.add_extra_dim(laspy.ExtraBytesParams(name="unrelated", type=np.uint32))
    base.unrelated = [10, 11, 12, 13, 14, 15]
    source = tmp_path / "base.laz"
    base.write(source)
    base.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance_SAT", type=np.int32))
    base.PredInstance_SAT = [-1, 3, 3, 8, 8, 8]
    enriched = tmp_path / "enriched.laz"
    base.write(enriched)
    return source, enriched, {"generation": str(uuid4()), "item": 1}


def save(cloud, path):
    _, enriched, identity = cloud
    return export(
        enriched,
        path,
        **identity,
        owned=["PredInstance_SAT"],
        producer={"fixture": True},
    )


def test_roundtrip_keeps_duplicate_coordinates_separate_and_preserves_all_base_fields(
    cloud, tmp_path
):
    source, enriched, identity = cloud
    sidecar, output = tmp_path / "sat.parquet", tmp_path / "assembled.laz"
    save(cloud, sidecar)
    # Reordering Parquet rows is valid: explicit indices define association.
    table = pq.read_table(sidecar)
    pq.write_table(table.take(pa.array([5, 4, 3, 2, 1, 0])), sidecar)
    assemble(source, [sidecar], output, **identity)
    actual = laspy.read(output)
    unchanged_base(laspy.read(source), actual)
    assert np.array_equal(
        actual.PredInstance_SAT, laspy.read(enriched).PredInstance_SAT
    )
    assert actual.PredInstance_SAT[0] != actual.PredInstance_SAT[1]


def test_sidecar_with_another_point_count_is_rejected(cloud, tmp_path):
    source, enriched, identity = cloud
    result = laspy.read(enriched)
    result.points = result.points[:5].copy()
    result.write(enriched)
    sidecar = tmp_path / "short.parquet"
    save(cloud, sidecar)
    with pytest.raises(ValueError, match="point count"):
        assemble(source, [sidecar], tmp_path / "bad.laz", **identity)


@pytest.mark.parametrize("indices", [[0, 0, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6]])
def test_missing_duplicate_and_out_of_range_indices_fail(cloud, tmp_path, indices):
    source, _, identity = cloud
    sidecar = tmp_path / "bad.parquet"
    save(cloud, sidecar)
    table = pq.read_table(sidecar)
    table = table.set_column(0, "point_index", pa.array(indices, type=pa.uint64()))
    pq.write_table(table, sidecar)
    with pytest.raises(ValueError, match="point indices"):
        assemble(source, [sidecar], tmp_path / "bad.laz", **identity)


@pytest.mark.parametrize("change", ["item", "generation"])
def test_wrong_base_identity_fails(cloud, tmp_path, change):
    source, _, identity = cloud
    sidecar = tmp_path / "sat.parquet"
    save(cloud, sidecar)
    identity[change] = 2 if change == "item" else str(uuid4())
    with pytest.raises(ValueError, match="base identity"):
        assemble(source, [sidecar], tmp_path / "bad.laz", **identity)


def test_owned_dimensions_cannot_overwrite_another_product(cloud, tmp_path):
    source, _, identity = cloud
    sidecar = tmp_path / "sat.parquet"
    save(cloud, sidecar)
    with pytest.raises(ValueError, match="ownership collision"):
        assemble(source, [sidecar, sidecar], tmp_path / "bad.laz", **identity)


def test_existing_artifacts_are_not_overwritten(cloud, tmp_path):
    source, _, identity = cloud
    sidecar = tmp_path / "sat.parquet"
    save(cloud, sidecar)
    with pytest.raises(ValueError, match="already exists"):
        save(cloud, sidecar)
    with pytest.raises(ValueError, match="must be new"):
        assemble(source, [sidecar], source, **identity)


def test_independent_methods_assemble_without_changing_base(cloud, tmp_path):
    source, _, identity = cloud
    sat = tmp_path / "sat.parquet"
    save(cloud, sat)
    base = laspy.read(source)
    base.add_extra_dim(laspy.ExtraBytesParams(name="species_id_SAT", type=np.uint16))
    base.species_id_SAT = [0, 2, 2, 7, 7, 7]
    species_laz = tmp_path / "species.laz"
    base.write(species_laz)
    species = tmp_path / "species.parquet"
    export(
        species_laz,
        species,
        **identity,
        owned=["species_id_SAT"],
        producer={"inputs": {"sat": "retained-sat-product"}},
    )
    output = tmp_path / "combined.laz"
    assemble(source, [sat, species], output, **identity)
    result = laspy.read(output)
    unchanged_base(laspy.read(source), result)
    assert np.array_equal(result.species_id_SAT, base.species_id_SAT)
    assert list(result.PredInstance_SAT) == [-1, 3, 3, 8, 8, 8]


def test_chunk_boundaries_preserve_reordered_rows_and_reject_cross_batch_duplicates(
    cloud, tmp_path, monkeypatch
):
    module = module_under_test

    monkeypatch.setattr(module, "CHUNK_POINTS", 2)
    source, enriched, identity = cloud
    path = tmp_path / "chunked.parquet"
    save(cloud, path)
    table = pq.read_table(path)
    pq.write_table(table.take(pa.array([5, 1, 4, 0, 3, 2])), path)
    output = tmp_path / "chunked.laz"
    assemble(source, [path], output, **identity)
    unchanged_base(laspy.read(enriched), laspy.read(output))
    table = table.set_column(
        0, "point_index", pa.array([0, 1, 2, 3, 0, 5], type=pa.uint64())
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="point indices"):
        assemble(source, [path], tmp_path / "duplicate.laz", **identity)


def test_sidecars_written_by_export_stream_without_index_scratch(
    cloud, tmp_path, monkeypatch
):
    module = module_under_test

    source, enriched, identity = cloud
    sidecar, output = tmp_path / "sat.parquet", tmp_path / "assembled.laz"
    save(cloud, sidecar)

    def no_indexed(*args, **kwargs):
        raise AssertionError("ordered sidecars must not need disk-backed index arrays")

    monkeypatch.setattr(module, "_assemble_indexed", no_indexed)
    monkeypatch.setattr(module, "CHUNK_POINTS", 2)  # Stream across chunks.
    assemble(source, [sidecar], output, **identity)
    assert np.array_equal(
        laspy.read(output).PredInstance_SAT, laspy.read(enriched).PredInstance_SAT
    )


def test_rows_that_only_look_ordered_fall_back_to_the_indexed_path(
    cloud, tmp_path, monkeypatch
):
    module = module_under_test

    source, enriched, identity = cloud
    sidecar, output = tmp_path / "sat.parquet", tmp_path / "assembled.laz"
    save(cloud, sidecar)
    # One row group with min 0 and max 5, but rows swapped: statistics pass,
    # streaming finds the swap and the indexed path assembles correctly.
    table = pq.read_table(sidecar)
    pq.write_table(table.take(pa.array([1, 0, 2, 3, 4, 5])), sidecar)
    calls = []
    indexed = module._assemble_indexed
    monkeypatch.setattr(
        module, "_assemble_indexed", lambda *a, **k: (calls.append(1), indexed(*a, **k))
    )
    assemble(source, [sidecar], output, **identity)
    assert calls == [1]
    assert np.array_equal(
        laspy.read(output).PredInstance_SAT, laspy.read(enriched).PredInstance_SAT
    )


def test_only_named_numeric_extra_dimensions_can_be_stored(cloud, tmp_path):
    _, enriched, identity = cloud
    for owned in (["intensity"], ["missing"], []):
        with pytest.raises(ValueError):
            export(
                enriched,
                tmp_path / f"bad-{len(owned)}-{owned}.parquet",
                **identity,
                owned=owned,
                producer={},
            )


@pytest.mark.parametrize(
    "dtype, scale",
    [(np.float32, 1.0), (np.float32, 0.01), (np.float64, 1.0), (np.int32, 0.001)],
)
def test_scaled_extra_bytes_keep_their_stored_values(tmp_path, dtype, scale):
    # Datasets 1919, 2030 and 3827: a float `Range` declared with a scale
    # came back rounded (2.93 -> 3) because it went through the scale.
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(
        laspy.ExtraBytesParams(
            name="Range", type=dtype, scales=np.array([scale]), offsets=np.array([0.0])
        )
    )
    base = laspy.LasData(header)
    base.X = np.arange(5)
    raw = np.array([2.93, 0.5302302, 17.25, -1.0, 30.7]) / (
        1 if np.dtype(dtype).kind == "f" else scale
    )
    base.points.array["Range"] = raw.astype(dtype)
    source, output = tmp_path / "base.laz", tmp_path / "assembled.laz"
    base.write(source)
    identity = {"generation": str(uuid4()), "item": 1}
    assemble(source, [], output, **identity)
    assert (
        laspy.read(output).points.array["Range"].tobytes()
        == laspy.read(source).points.array["Range"].tobytes()
    )


def test_a_rounded_scaled_extra_byte_is_still_rejected(tmp_path):
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(
        laspy.ExtraBytesParams(
            name="Range",
            type=np.float32,
            scales=np.array([1.0]),
            offsets=np.array([0.0]),
        )
    )
    base = laspy.LasData(header)
    base.X = np.arange(3)
    base.points.array["Range"] = np.array([2.93, 4.2, 5.5], dtype=np.float32)
    source, member = tmp_path / "base.laz", tmp_path / "member.laz"
    base.write(source)
    base.points.array["Range"] = np.array([3.0, 4.2, 5.5], dtype=np.float32)
    base.write(member)
    with pytest.raises(ValueError, match="dimension changed: Range"):
        module_under_test.unchanged_files(source, member)


def test_a_member_with_the_same_final_values_in_another_storage_is_accepted(tmp_path):
    # The check compares final values (stored x scale + offset), as decided by
    # Kilian on 8 Nov 2026: 293 at scale 0.01 equals 2930 at scale 0.001.
    paths = {}
    for name, dtype, scale, stored in (
        ("base", np.uint16, 0.01, [293, 420, 550]),
        ("member", np.int32, 0.001, [2930, 4200, 5500]),
    ):
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="Range",
                type=dtype,
                scales=np.array([scale]),
                offsets=np.array([0.0]),
            )
        )
        cloud = laspy.LasData(header)
        cloud.X = np.arange(3)
        cloud.points.array["Range"] = np.array(stored, dtype=dtype)
        paths[name] = tmp_path / f"{name}.laz"
        cloud.write(paths[name])
    module_under_test.unchanged_files(paths["base"], paths["member"])


def test_assembly_keeps_the_method_output_descriptions(cloud, tmp_path):
    source, enriched, identity = cloud
    method = laspy.read(source)
    method.add_extra_dim(
        laspy.ExtraBytesParams(
            name="PredInstance_FM", type=np.uint16, description="ForestMamba instance"
        )
    )
    method.PredInstance_FM = [0, 1, 1, 2, 2, 2]
    method.write(enriched)
    sidecar, output = tmp_path / "fm.parquet", tmp_path / "assembled.laz"
    export(enriched, sidecar, **identity, owned=["PredInstance_FM"], producer={})
    assemble(source, [sidecar], output, **identity)
    dimension = laspy.read(output).point_format.dimension_by_name("PredInstance_FM")
    assert dimension.description == "ForestMamba instance"
    # The base keeps its own descriptions through the header copy.
    assert laspy.read(output).point_format.dimension_by_name(
        "unrelated"
    ).description == (
        laspy.read(source).point_format.dimension_by_name("unrelated").description
    )


def test_viewer_merge_keeps_scaled_extra_bytes_as_stored(tmp_path):
    paths = []
    for number, values in enumerate(([2.93, 0.53], [17.25, -1.0])):
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.offsets = [400000 + number, 5300000, 0]
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="Range",
                type=np.float32,
                scales=np.array([1.0]),
                offsets=np.array([0.0]),
            )
        )
        cloud = laspy.LasData(header)
        cloud.X = [0, 1000]
        cloud.points.array["Range"] = np.array(values, dtype=np.float32)
        paths.append(tmp_path / f"part{number}.laz")
        cloud.write(paths[-1])
    output = tmp_path / "viewer.laz"
    module_under_test.merge_for_viewer(paths, output)
    merged = laspy.read(output)
    assert (
        merged.points.array["Range"].tolist()
        == np.array([2.93, 0.53, 17.25, -1.0], dtype=np.float32).tolist()
    )
    assert np.allclose(merged.x, [400000, 400010, 400001, 400011])
