# Copilot instructions for `wire-doodler`

## Project focus
- This repo experiments with wire-geometry electromagnetic modeling primitives (early-stage, numerical geometry + quadrature core).
- The public API is re-exported from `doodler/__init__.py`; prefer importing from `doodler` in tests/examples.

## Architecture you should understand first
- `doodler/geometry/` defines surfaces in a normalized tangent domain `(s,t) ∈ [-1,1] × [-1,1]`.
  - Base contract: `Shape3D` in `doodler/geometry/common.py`.
  - Implementations: `Cylinder`, `ClippedSphere`.
- `doodler/quadrature/` provides cached integration rules.
  - `RuleCache` loads various 1D rules from bundled parquet files and builds 2D tensor rules (`Rule2D`).
- `doodler/geometry/sampler.py` bridges both: `Shape3DSampler` accesses rule sizes from geometric spans, then maps 2D rule positions to 3D points.

## Critical conventions (project-specific)
- Numeric dtypes are explicit aliases (`Real`, `Index`, `Integer`), mostly `numpy.float64`/`int64`; preserve these when adding arrays.
- Validate and copy vector inputs through `r3vector_copy` / `axes3d_copy` (`doodler/r3.py`) instead of trusting caller arrays.
- Error taxonomy is meaningful:
  - raise `Recoverable` for retriable conditions (e.g., absent cached rule files),
  - raise `Unrecoverable` for invalid state,
  - use `NeverImplement` for immutable setters/abstract behavior.
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

## Data/dependency integration points
- Quadrature rule files are local assets in `doodler/quadrature/*.parquet`; `RuleCache` resolves them relative to that package directory.
- Rule generation script exists at `doodler/quadrature/generate_quadrature_rules.jl` (Julia), but normal Python workflows consume existing parquet files.
- Sampling accuracy behavior is validated against SciPy integration in tests; changing span/rule selection logic in `Shape3DSampler` will affect many assertions.

## Testing patterns to mirror
- Tests use tolerance-based assertions (`real_equality`, `r3vector_equality`) rather than exact equality for floating-point math.
- Randomized geometry tests are deterministic via a fixed NumPy RNG seed in `tests/test_shape_samples.py`.
- When adding geometry features, include both aligned and rotated-frame cases (see cylinder/sphere area tests).
