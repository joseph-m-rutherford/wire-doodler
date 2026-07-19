# Copilot instructions for `wire-doodler`

## Project focus
- This repo experiments with wire-geometry electromagnetic modeling primitives (early-stage, numerical geometry + quadrature core).
- The public API is re-exported from `doodler/__init__.py`; prefer importing from `doodler` in tests/examples.

## Architecture you should understand first
- `doodler/common.py` defines scalar/index aliases (`Real`, `Index`, `Integer`) used consistently across modules.
- `doodler/errors.py` defines the project error taxonomy (`Recoverable`, `Unrecoverable`, `NeverImplement`, `NotYetImplemented`).
- `doodler/r3.py` defines 3-D vector/axes validation and tolerant equality helpers (e.g., `vector_copy`, `axes3d_copy`, `vector_equality`).
- `doodler/geometry/` contains geometry primitives and 2-D wire segment containers:
  - `geometry/common.py`: `Shape3D` base contract + tangent-domain validation.
  - `geometry/cylinder.py`, `geometry/clipped_sphere.py`: concrete `Shape3D` implementations.
  - `geometry/sampler.py`: `Shape3DSampler` maps quadrature points `(s,t)` to 3-D geometry points.
  - `geometry/wire_segments.py`: `WireSegment2D` immutable 2-D segment data + `as_xyz` transform from local UV to global XYZ.
- `doodler/operators/` contains wire-operator assembly logic:
  - `operators/wire_mesh.py`: `WireMesh3D` mesh construction, collision/intersection checks, and flat subsegment indexing.
  - `operators/mesh_functions.py`: `MeshFunctions` mapping from function index to subsegment-pair support.
  - `operators/fillers.py`: `WireMesh3DFill` and `FillChoice` for mass/stiffness local assembly over overlapping mesh support.
- `doodler/io_formats/` handles external formats:
  - `io_formats/svg_reader.py`: parses SVG `<line>`, `<polyline>`, `<path>` into `WireSegment2D` objects with required `<desc>` metadata.
  - `io_formats/vtk_writer.py`: exports wire/polyline geometry to VTK PolyData format.
- `doodler/quadrature/` provides cached integration rules:
  - `quadrature/cached_rules.py`: `RuleCache` loads 1D rules and forms tensor-product 2D rules.
  - `quadrature/rules.py`: quadrature rule data containers/utilities.
  - `quadrature/generate_quadrature_rules.jl`: source generator for bundled parquet rule files.
- Public API is re-exported from `doodler/__init__.py`.

## Critical conventions (project-specific)
- Numeric dtypes are explicit aliases (`Real`, `Index`, `Integer`), mostly `numpy.float64`/`int64`; preserve these when adding arrays.
- Validate and copy vector inputs through `vector_copy` / `axes3d_copy` (`doodler/r3.py`) instead of trusting caller arrays.
- Error taxonomy is meaningful:
  - raise `Recoverable` for retriable conditions (e.g., absent cached rule files),
  - raise `Unrecoverable` for invalid state,
  - use `NeverImplement` for immutable setters/abstract behavior.
- Member variables use a leading `_` (e.g., `self._points`); expose them via a `@property` getter (no leading `_`) and a setter that raises `NeverImplement` to enforce immutability.
- Many properties are intentionally immutable and use setter methods that always raise; do not add mutability unless tests demand it.
- Coordinate guardrails are strict: geometry methods must reject out-of-range tangent coordinates via `InvalidTangentCoordinates`.

## Workflows and commands
- Python dependencies are listed in `requirements-python.txt`; install them with:
  - `python -m pip install -r requirements-python.txt`
  - If you prefer manual installs, the equivalent single-package command is:
    - `python -m pip install numpy pyarrow scipy pytest`
- Run all tests:
  - `python -m pytest -q`
- Run targeted tests while iterating geometry/sampling:
  - `python -m pytest tests/test_geometry.py tests/test_shape_samples.py -q`
- Run targeted tests for wire operators and segment parsing:
  - `python -m pytest tests/test_wire_mesh_3d.py tests/test_fillers.py tests/test_wire_segments.py tests/test_svg.py -q`

## Data/dependency integration points
- Quadrature rules are trivially computed or obtained from the `modepy` package
- Sampling accuracy behavior is validated against SciPy integration in tests; changing span/rule selection logic in `Shape3DSampler` will affect many assertions.

## Testing patterns to mirror
- Tests use tolerance-based assertions (`real_equality`, `vector_equality`) rather than exact equality for floating-point math.
- Randomized geometry tests are deterministic via a fixed NumPy RNG seed in `tests/test_shape_samples.py`.
- When adding geometry features, include both aligned and rotated-frame cases (see cylinder/sphere area tests).
- Wire mesh/operator tests are split by responsibility:
  - `tests/test_wire_segments.py` focuses on `WireSegment2D` immutability/container behavior.
  - `tests/test_wire_mesh_3d.py` focuses on `WireMesh3D` construction, collision guards, indexing, and immutability.
  - `tests/test_fillers.py` focuses on `WireMesh3DFill` mass/stiffness overlap behavior.

**Module Dependency Flow**
- **SVG → 2D → 3D:** `io_formats/svg_reader.py` parses SVG into `WireSegment2D`; `geometry.wire_segments.as_xyz` maps (u,v) in a supplied `uvw` frame + `xyz_offset` to global 3‑D points.
- **Wire mesh pipeline:** `operators/wire_mesh.py` builds `WireMesh3D` from named polylines (output of `as_xyz`); it constructs the flat `subsegment_index` and uses `operators.mesh_functions.MeshFunctions` to enumerate function support pairs.
- **Local assembly:** `operators/fillers.py` (`WireMesh3DFill`) consumes `WireMesh3D` and `MeshFunctions` to produce mass/stiffness filler callables for overlapping function supports.
- **Geometry sampling:** `geometry.sampler.Shape3DSampler` selects quadrature sizes from `quadrature.RuleCache` and maps 2‑D rule points to 3‑D coordinates on `Shape3D` implementations (cylinder, clipped sphere).
- **Public surface:** `doodler/__init__.py` re-exports the main user-facing symbols (`as_xyz`, `WireMesh3D`, `MeshFunctions`, `WireMesh3DFill`, `FillChoice`, geometry primitives, and quadrature helpers).
