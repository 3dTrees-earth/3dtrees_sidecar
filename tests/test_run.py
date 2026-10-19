"""The Galaxy sidecar tool's CLI on small fixtures: export, assemble, check."""

import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import laspy
import numpy as np
import pyarrow.parquet as pq
import pytest
from pyproj import CRS

RUN = Path(__file__).parents[1] / "src/run.py"
spec = importlib.util.spec_from_file_location("sidecar_tool", RUN)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

SAT = {"PredInstance_SAT": np.uint32, "PredSemantic_SAT": np.uint32}
FM = {
    "PredInstance_FM": np.uint32,
    "PredSemantic_FM": np.uint32,
    "PredScore_FM": np.float32,
}
SPECIES_SAT = {"species_id_SAT": np.uint32, "species_prob_SAT": np.float32}


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    # Production steps are a million points; smaller ones keep every command
    # crossing chunk boundaries on these fixtures.
    monkeypatch.setattr(tool.sidecar, "CHUNK_POINTS", 100_000)


def write(path, base, columns, seed):
    cloud = laspy.LasData(base.header.copy())
    cloud.points = base.points.copy()
    rng = np.random.default_rng(seed)
    for name, dtype in columns.items():
        cloud.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        cloud[name] = rng.integers(0, 50, len(base.points)).astype(dtype)
    cloud.write(path)
    return path


@pytest.fixture
def item(tmp_path):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(32632))
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [400000, 5300000, 200]
    base = laspy.LasData(header)
    count = 250_000  # Several chunks of the 100,000 points set above.
    rng = np.random.default_rng(1)
    base.X = rng.integers(0, 10_000, count)
    base.Y = rng.integers(0, 10_000, count)
    base.Z = rng.integers(0, 3_000, count)
    base.intensity = rng.integers(0, 255, count)
    source = tmp_path / "base.laz"
    base.write(source)
    sat = write(tmp_path / "sat.laz", base, SAT, 2)
    fm = write(tmp_path / "fm.laz", base, FM, 3)
    # Species output repeats its segmentation's labels next to its own columns.
    species = laspy.read(sat)
    rng = np.random.default_rng(4)
    for name, dtype in SPECIES_SAT.items():
        species.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        species[name] = rng.integers(0, 9, count).astype(dtype)
    species.write(tmp_path / "species_sat.laz")
    return {
        "dir": tmp_path,
        "base": source,
        "sat": sat,
        "forestmamba": fm,
        "species_sat": tmp_path / "species_sat.laz",
        "generation": str(uuid4()),
        "item": "11353",
    }


def run(*argv):
    return tool.main([str(a) for a in argv])


def identity(item):
    return [
        "--base",
        item["base"],
        "--generation",
        item["generation"],
        "--item",
        item["item"],
    ]


def export(item, kind, carried=None):
    out = item["dir"] / f"{kind}.parquet"
    args = ["export", *identity(item), "--enriched", item[kind], "--output", out]
    args += ["--report", item["dir"] / f"{kind}.json"]
    if carried:
        args += ["--carried", carried]
    assert run(*args) == 0
    return out, json.loads((item["dir"] / f"{kind}.json").read_text())


def assemble(item, sidecars, name="member"):
    out = item["dir"] / f"{name}.laz"
    args = ["assemble", *identity(item), "--output", out]
    args += ["--report", item["dir"] / f"{name}.json"]
    for path in sidecars:
        args += ["--sidecar", path]
    assert run(*args) == 0
    return out, json.loads((item["dir"] / f"{name}.json").read_text())


def check(item, sidecars, member):
    args = ["check", *identity(item), "--member", member]
    args += ["--report", item["dir"] / "check.json"]
    for path in sidecars:
        args += ["--sidecar", path]
    return run(*args)


def test_export_derives_owned_columns_and_reports_identity(item):
    sat, report = export(item, "sat")
    assert sorted(report["dimensions"]) == sorted(SAT)
    assert report["item_id"] == 11353
    assert report["generation"] == item["generation"]
    assert report["sidecar_sha256"] == tool.sidecar.sha256(sat)
    assert report["producer"]["tool"] == "3dtrees_sidecar"


def test_species_export_drops_the_carried_segmentation_columns(item):
    sat, _ = export(item, "sat")
    _, report = export(item, "species_sat", carried=sat)
    assert sorted(report["dimensions"]) == sorted(SPECIES_SAT)


@pytest.mark.parametrize(
    "kinds",
    [[], ["sat"], ["sat", "forestmamba"]],
    ids=["points-only", "one-segmentation", "both-segmentations"],
)
def test_assemble_and_check_every_viewer_case(item, kinds):
    sidecars = [export(item, kind)[0] for kind in kinds]
    member, report = assemble(item, sidecars)
    expected = sorted(n for kind in kinds for n in (SAT if kind == "sat" else FM))
    assert report["dimensions"] == expected
    assert report["points"] == 250_000
    actual = laspy.read(member)
    for kind in kinds:
        enriched = laspy.read(item[kind])
        for name in SAT if kind == "sat" else FM:
            assert np.array_equal(actual[name], enriched[name])
    assert check(item, sidecars, member) == 0
    result = json.loads((item["dir"] / "check.json").read_text())
    assert result["member_sha256"] == report["output_sha256"]
    assert result["points"] == 250_000


def test_check_rejects_a_member_with_a_changed_value(item):
    sat, _ = export(item, "sat")
    member, _ = assemble(item, [sat])
    tampered = laspy.read(member)
    tampered.PredInstance_SAT[123_456] += 1
    tampered.write(member)
    assert check(item, [sat], member) == 2


def test_check_rejects_a_member_missing_a_selected_sidecar(item):
    sat, _ = export(item, "sat")
    fm, _ = export(item, "forestmamba")
    member, _ = assemble(item, [sat])
    assert check(item, [sat, fm], member) == 2


def test_assemble_rejects_a_sidecar_from_another_generation(item):
    sat, _ = export(item, "sat")
    item["generation"] = str(uuid4())
    out = item["dir"] / "member.laz"
    args = ["assemble", *identity(item), "--output", out, "--sidecar", sat]
    assert run(*args, "--report", item["dir"] / "r.json") == 2


def test_item_must_be_a_dataset_item_id(item):
    item["item"] = "not-an-id"
    with pytest.raises(SystemExit):
        assemble(item, [])


def test_columns_split_a_combined_legacy_file_into_one_sidecar_per_method(item):
    # A legacy analysis carries SAT and ForestMamba columns in one file.
    combined = laspy.read(item["sat"])
    fm = laspy.read(item["forestmamba"])
    for name, dtype in FM.items():
        combined.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        combined[name] = fm[name]
    item["legacy"] = item["dir"] / "legacy.laz"
    combined.write(item["legacy"])
    sidecars = []
    for kind, columns in (("sat", SAT), ("forestmamba", FM)):
        out = item["dir"] / f"legacy-{kind}.parquet"
        args = ["export", *identity(item), "--enriched", item["legacy"]]
        for name in columns:
            args += ["--column", name]
        args += ["--output", out]
        assert run(*args, "--report", item["dir"] / f"legacy-{kind}.json") == 0
        assert tool.manifest_of(out)["dimensions"].keys() == columns.keys()
        sidecars.append(out)
    member, _ = assemble(item, sidecars)
    actual = laspy.read(member)
    for name in [*SAT, *FM]:
        assert np.array_equal(actual[name], combined[name])
    assert check(item, sidecars, member) == 0


def test_columns_must_exist_in_the_output(item):
    out = item["dir"] / "x.parquet"
    args = ["export", *identity(item), "--enriched", item["sat"]]
    args += ["--column", "PredInstance_FM", "--output", out]
    assert run(*args, "--report", item["dir"] / "x.json") == 2


def test_sidecars_store_the_index_delta_encoded_in_large_row_groups(item):
    sat, _ = export(item, "sat")
    metadata = pq.ParquetFile(sat).metadata
    assert metadata.num_row_groups == 1  # 250,000 points, one group
    index = metadata.row_group(0).column(0)
    assert index.path_in_schema == "point_index"
    assert "DELTA_BINARY_PACKED" in index.encodings


def described(item):
    # A method output that labels its columns in the LAS extra-bytes records.
    cloud = laspy.read(item["base"])
    rng = np.random.default_rng(5)
    cloud.add_extra_dim(
        laspy.ExtraBytesParams(
            "PredInstance_SAT", np.uint32, description="Tree from file", no_data=[0]
        )
    )
    cloud.add_extra_dim(laspy.ExtraBytesParams("PredSemantic_SAT", np.uint32))
    cloud.PredInstance_SAT = rng.integers(0, 50, len(cloud.points)).astype(np.uint32)
    cloud.PredSemantic_SAT = rng.integers(0, 5, len(cloud.points)).astype(np.uint32)
    item["described"] = item["dir"] / "described.laz"
    cloud.write(item["described"])
    return item["described"]


def test_file_attributes_reach_the_member_unless_set_manually(item):
    out = item["dir"] / "attributed.parquet"
    args = ["export", *identity(item), "--enriched", described(item)]
    args += ["--column", "PredInstance_SAT", "--no-data", "4294967295"]
    args += ["--column", "PredSemantic_SAT", "--description", "Semantic class"]
    args += ["--no-data", "-0"]
    assert run(*args, "--output", out, "--report", item["dir"] / "a.json") == 0
    assert tool.manifest_of(out)["attributes"] == {
        "PredInstance_SAT": {"description": "Tree from file", "no_data": 4294967295},
        "PredSemantic_SAT": {"description": "Semantic class", "no_data": 0},
    }
    member, _ = assemble(item, [out])
    assert tool.sidecar.stored_attributes(member) == tool.manifest_of(out)["attributes"]
    assert check(item, [out], member) == 0


def test_check_rejects_a_member_with_changed_attributes(item):
    out = item["dir"] / "attributed.parquet"
    args = ["export", *identity(item), "--enriched", described(item)]
    assert run(*args, "--output", out, "--report", item["dir"] / "a.json") == 0
    member, _ = assemble(item, [out])
    # Same values, but the manifest no longer carries the attributes.
    table = pq.read_table(out)
    manifest = tool.manifest_of(out)
    manifest["attributes"] = {}
    stripped = item["dir"] / "stripped.parquet"
    pq.write_table(
        table.replace_schema_metadata({b"3dtrees": json.dumps(manifest).encode()}),
        stripped,
    )
    plain, _ = assemble(item, [stripped], name="plain")
    assert check(item, [stripped], plain) == 0
    assert check(item, [out], plain) == 2
    assert check(item, [out], member) == 0


@pytest.mark.parametrize(
    "attribute",
    [
        ["--no-data", "-1"],
        ["--no-data", "1.5"],
        ["--no-data", "none"],
        ["--description", "x" * 33],
        ["--description", "Baumhöhe"],
    ],
)
def test_attributes_that_las_cannot_store_are_rejected(item, attribute):
    out = item["dir"] / "bad.parquet"
    args = ["export", *identity(item), "--enriched", item["sat"]]
    args += ["--column", "PredInstance_SAT", *attribute, "--column", "PredSemantic_SAT"]
    assert run(*args, "--output", out, "--report", item["dir"] / "bad.json") == 2
    assert not out.exists()


def test_attributes_need_a_preceding_column(item):
    args = ["export", *identity(item), "--enriched", item["sat"], "--no-data", "0"]
    with pytest.raises(SystemExit):
        run(
            *args,
            "--output",
            item["dir"] / "x.parquet",
            "--report",
            item["dir"] / "x.json",
        )


def test_float_no_data_is_stored_as_its_float32_value(item):
    out = item["dir"] / "fm.parquet"
    args = ["export", *identity(item), "--enriched", item["forestmamba"]]
    args += ["--column", "PredScore_FM", "--no-data", "-1.5"]
    args += ["--column", "PredInstance_FM", "--column", "PredSemantic_FM"]
    assert run(*args, "--output", out, "--report", item["dir"] / "fm.json") == 0
    assert tool.manifest_of(out)["attributes"] == {"PredScore_FM": {"no_data": -1.5}}
    member, _ = assemble(item, [out])
    assert tool.sidecar.stored_attributes(member)["PredScore_FM"] == {"no_data": -1.5}
