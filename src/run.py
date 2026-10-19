"""Indexed scientific sidecars as Galaxy jobs: export, assemble and check.

One item per call; Galaxy maps the tool over linked collections whose element
identifiers are the Dataset item ids. Every subcommand writes one JSON report
next to its product so the native worker can verify the published bytes
(hashes, sizes, point counts, dimensions) without downloading point clouds.

  export    base + method-enriched LAZ (+ carried parent sidecar) -> sidecar
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
VERSION = "0.2.0"
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


class Column(argparse.Action):
    """`--column` starts a kept dimension; `--description` and `--no-data`
    set an attribute of the column named just before them."""

    def __call__(self, parser, namespace, value, option_string=None):
        columns = getattr(namespace, self.dest) or []
        if self.const is None:
            columns.append({"name": value})
        elif not columns:
            parser.error(f"{option_string} must follow a --column")
        else:
            columns[-1][self.const] = value
        setattr(namespace, self.dest, columns)


def manifest_of(path):
    """The identity and dimensions a sidecar carries in its Parquet metadata."""
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return json.loads(metadata[b"3dtrees"])


def write_report(path, document):
    Path(path).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")


def added_dimensions(base, enriched, carried):
    """The enriched file's new dimensions minus those its parent owns."""
    import laspy

    with laspy.open(base) as a, laspy.open(enriched) as b:
        added = set(b.header.point_format.dimension_names) - set(
            a.header.point_format.dimension_names
        )
    parent = set(manifest_of(carried)["dimensions"]) if carried else set()
    return sorted(added - parent)


def export(args):
    added = added_dimensions(args.base, args.enriched, args.carried)
    # Naming columns splits a combined file (a legacy analysis holding SAT and
    # ForestMamba) into one sidecar per method; the rest is left to its own.
    columns = args.columns or []
    owned = sorted(c["name"] for c in columns) if columns else added
    if len(set(owned)) != len(owned):
        raise ValueError("A column is named more than once")
    missing = set(owned) - set(added)
    if missing:
        raise ValueError(f"Requested columns are not in the output: {sorted(missing)}")
    attributes = {
        c["name"]: {k: v for k, v in c.items() if k != "name"} for c in columns
    }
    manifest = sidecar.export(
        args.base,
        args.enriched,
        args.output,
        generation=args.generation,
        item=args.item,
        owned=owned,
        carried=args.carried,
        ignored=sorted(set(added) - set(owned)),
        attributes=attributes,
        producer={"executor": "galaxy", "tool": TOOL, "version": VERSION},
    )
    write_report(args.report, {"item_id": args.item, **manifest})


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
        attributes = sidecar.stored_attributes(expected)
        if sidecar.stored_attributes(args.member) != attributes:
            raise ValueError("Member column descriptions or no-data values differ")
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

    def common(command):
        command.add_argument("--base", required=True, help="Standard base LAZ")
        command.add_argument("--generation", required=True, type=generation_id)
        command.add_argument("--item", required=True, type=item_id)
        command.add_argument("--report", required=True, help="JSON report path")

    one = commands.add_parser("export", help="Write one item's sidecar")
    common(one)
    one.add_argument("--enriched", required=True, help="Method output LAZ")
    one.add_argument("--carried", help="Parent sidecar the method output repeats")
    one.add_argument("--output", required=True, help="Sidecar Parquet path")
    one.add_argument(
        "--column",
        dest="columns",
        action=Column,
        metavar="NAME",
        help="Dimension to keep; repeat for each (default: every added one)",
    )
    one.add_argument(
        "--description",
        dest="columns",
        action=Column,
        const="description",
        metavar="TEXT",
        help="Description of the preceding --column (ASCII, at most 32 characters)",
    )
    one.add_argument(
        "--no-data",
        dest="columns",
        action=Column,
        const="no_data",
        metavar="VALUE",
        help="No-data value of the preceding --column",
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
