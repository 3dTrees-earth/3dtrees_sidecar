# 3Dtrees Sidecar

Galaxy tool for the 3Dtrees.earth processing pipeline. It moves the point-cloud
work that follows the scientific models into Galaxy: storing each method's
results as a small indexed column file ("sidecar"), building the downloadable
point cloud, and checking it independently.

| Task | Inputs (one item) | Outputs |
| --- | --- | --- |
| `export` | standard base LAZ, method output LAZ, optional parent sidecar, optional columns with attributes | sidecar Parquet, JSON report |
| `assemble` | standard base LAZ, any selection of sidecars (SAT, ForestMamba, species) | downloadable LAZ, JSON report |
| `check` | the published base, sidecars and downloadable LAZ | JSON check report |

- **export** writes `point_index` plus only the dimensions the method added, and
  refuses outputs whose base points or point order changed. Naming columns with
  `--column` splits a combined file (a legacy analysis with several methods) into
  one sidecar per method.
- Each column keeps the description and no-data value the method output stores
  in its LAS extra-bytes record. `--description` and `--no-data` after a
  `--column` set them by hand instead. They travel in the sidecar's metadata and
  reach the assembled LAZ, where `check` compares them too.
- **assemble** keeps every base field unchanged and in order and adds the
  selected sidecars' columns. Any subset is valid (points only, one method, all).
  Sidecars in point order stream in constant memory.
- **check** rebuilds the point cloud from base and sidecars and compares every
  dimension point by point.

Every task writes a JSON report with identities, point count, dimensions and
SHA-256 hashes, so the 3Dtrees worker verifies stored bytes without downloading
point clouds. On 85 M points (10 cores): export 20 s, assemble 35 s, check 62 s,
each under 0.6 GB of memory and without temporary disk.

## Usage

```sh
python src/run.py export   --base base.laz --generation <uuid> --item <id> --enriched sat.laz --output sat.parquet --report sat.json
python src/run.py export   --base base.laz --generation <uuid> --item <id> --enriched sat.laz \
  --column PredInstance_SAT --description "Tree instance" \
  --column PredSemantic_SAT --description "Semantic class" --output sat.parquet --report sat.json
python src/run.py assemble --base base.laz --generation <uuid> --item <id> --sidecar sat.parquet --output member.laz --report member.json
python src/run.py check    --base base.laz --generation <uuid> --item <id> --sidecar sat.parquet --member member.laz --report check.json
```

## Development

```sh
pip install -r requirements-dev.txt
python -m pytest -q tests
docker build -t ghcr.io/3dtrees-earth/3dtrees_sidecar:dev .
```

## Release

Bump `VERSION` in `src/run.py`, merge, and publish a GitHub release `v<version>`.
The release workflow checks the tag against `VERSION` and pushes
`ghcr.io/3dtrees-earth/3dtrees_sidecar:<version>`. The Galaxy wrapper lives in
[galaxytools](https://github.com/bgruening/galaxytools) under
`tools/3dtrees_sidecar` and pins the same version.
