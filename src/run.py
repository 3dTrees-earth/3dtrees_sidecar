"""Indexed scientific sidecars as Galaxy jobs: export, assemble and check.

One item per call; Galaxy maps the tool over linked collections whose element
identifiers are the Dataset item ids. Every subcommand writes one JSON report
next to its product so the native worker can verify the published bytes
(hashes, sizes, point counts, dimensions) without downloading point clouds.

  export    method output LAZ + the columns to keep -> sidecar
  assemble  base + selected sidecars -> downloadable member LAZ
  check     re-assemble independently and compare against a published member
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from uuid import UUID

import sidecar

TOOL = "3dtrees_sidecar"
VERSION = "0.1.0"
CHECK_CONTRACT = "3dtrees-sidecar-check-v1"


def item_id(value):
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError("item must be a positive Dataset item id")
    return int(value)


def generation_id(value):
    try:
        return str(UUID(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("generation must be a UUID") from error


def manifest_of(path):
    """The identity and dimensions a sidecar carries in its Parquet metadata."""
    import pyarrow.parquet as pq

    return sidecar.manifest_of(pq.ParquetFile(path))


def write_report(path, document):
    Path(path).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")


def export(args):
    import laspy

    with laspy.open(args.enriched) as enriched:
        point_format = enriched.header.point_format
        names = set(point_format.dimension_names)
        extra = set(point_format.extra_dimension_names)
    owned = sorted(args.columns)
    missing = set(owned) - names
    if missing:
        raise ValueError(f"Requested columns are not in the output: {sorted(missing)}")
    # Only the named columns are stored; any other added column is reported.
    dropped = sorted(extra - set(owned))
    if dropped:
        print(f"export warning: columns not kept: {dropped}", file=sys.stderr)
    manifest = sidecar.export(
        args.enriched,
        args.output,
        generation=args.generation,
        item=args.item,
        owned=owned,
        producer={"executor": "galaxy", "tool": TOOL, "version": VERSION},
    )
    write_report(args.report, {"item_id": args.item, **manifest, "dropped": dropped})


def assemble(args):
    result = sidecar.assemble(
        args.base,
        args.sidecar,
        args.output,
        generation=args.generation,
        item=args.item,
    )
    write_report(
        args.report,
        {
            "item_id": args.item,
            "file": f"{args.item}.laz",
            "bytes": Path(args.output).stat().st_size,
            "dimensions": sorted(
                name
                for path in args.sidecar
                for name in manifest_of(path)["dimensions"]
            ),
            **result,
        },
    )


def check(args):
    """Rebuild the member from the published inputs and compare it byte-for-value.

    Runs as its own job after publication, so it reads what was written rather
    than what assembly held in memory."""
    import laspy

    with tempfile.TemporaryDirectory(prefix="3dt-sidecar-check-", dir=".") as name:
        expected = Path(name) / "expected.laz"
        sidecar.assemble(
            args.base,
            args.sidecar,
            expected,
            generation=args.generation,
            item=args.item,
        )
        with laspy.open(args.member) as actual, laspy.open(expected) as reference:
            dimensions = set(actual.header.point_format.dimension_names)
            if dimensions != set(reference.header.point_format.dimension_names):
                raise ValueError("Member dimensions differ from the selected sidecars")
            points = actual.header.point_count
            if points != reference.header.point_count:
                raise ValueError("Member point count differs")
        # Every dimension of the rebuilt file (base fields and each selected
        # scientific column), point by point and in order.
        sidecar.unchanged_files(expected, args.member)
    write_report(
        args.report,
        {
            "contract": CHECK_CONTRACT,
            "item_id": args.item,
            "generation": args.generation,
            "base_sha256": sidecar.sha256(args.base),
            "sidecars": [
                {
                    "sha256": sidecar.sha256(path),
                    "dimensions": sorted(manifest_of(path)["dimensions"]),
                }
                for path in args.sidecar
            ],
            "member_sha256": sidecar.sha256(args.member),
            "member_bytes": Path(args.member).stat().st_size,
            "points": points,
            "checker": {"tool": TOOL, "version": VERSION},
        },
    )


def parser():
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root.add_argument("--version", action="version", version=VERSION)
    commands = root.add_subparsers(dest="command", required=True)

    def common(command, base=True):
        if base:
            command.add_argument("--base", required=True, help="Standard base LAZ")
        command.add_argument("--generation", required=True, type=generation_id)
        command.add_argument("--item", required=True, type=item_id)
        command.add_argument("--report", required=True, help="JSON report path")

    one = commands.add_parser("export", help="Write one item's sidecar")
    common(one, base=False)
    one.add_argument("--enriched", required=True, help="Method output LAZ")
    one.add_argument("--output", required=True, help="Sidecar Parquet path")
    one.add_argument(
        "--columns",
        required=True,
        type=lambda value: [c for c in value.split(",") if c],
        help="Comma-separated dimensions to store",
    )
    one.set_defaults(handler=export)

    two = commands.add_parser("assemble", help="Join sidecars onto the base")
    common(two)
    two.add_argument("--sidecar", action="append", default=[])
    two.add_argument("--output", required=True, help="Member LAZ path")
    two.set_defaults(handler=assemble)

    three = commands.add_parser("check", help="Independently verify a member")
    common(three)
    three.add_argument("--sidecar", action="append", default=[])
    three.add_argument("--member", required=True, help="Published member LAZ")
    three.set_defaults(handler=check)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        args.handler(args)
    except ValueError as error:
        print(f"{args.command} rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
