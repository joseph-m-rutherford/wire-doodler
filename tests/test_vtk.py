#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import numpy as np
import pytest

from doodler import as_xyz, export_polylines, Real, Unrecoverable, WireSegment2D, r3vector_equality
from doodler.r3 import TOLERANCE


# Standard basis: u=x, v=y, w=z
_IDENTITY_FRAME = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=Real)
_ZERO_OFFSET = np.array([0, 0, 0], dtype=Real)


def test_as_xyz_identity_frame_line():
    """With identity frame and zero offset, x,y map directly to x,y with z=0."""
    segs = {'seg1': WireSegment2D([(Real(0), Real(0)), (Real(10), Real(20))], 'segment one')}
    result = as_xyz(segs, _IDENTITY_FRAME, _ZERO_OFFSET)
    pts = result['seg1']
    assert len(pts) == 2
    assert r3vector_equality(pts[0], np.array([0, 0, 0], dtype=Real), TOLERANCE)
    assert r3vector_equality(pts[1], np.array([10, 20, 0], dtype=Real), TOLERANCE)


def test_as_xyz_identity_frame_polyline():
    segs = {'tri': WireSegment2D([(Real(0), Real(0)), (Real(5), Real(10)), (Real(10), Real(0))], 'triangle')}
    result = as_xyz(segs, _IDENTITY_FRAME, _ZERO_OFFSET)
    pts = result['tri']
    assert len(pts) == 3
    assert r3vector_equality(pts[0], np.array([0, 0, 0], dtype=Real), TOLERANCE)
    assert r3vector_equality(pts[1], np.array([5, 10, 0], dtype=Real), TOLERANCE)
    assert r3vector_equality(pts[2], np.array([10, 0, 0], dtype=Real), TOLERANCE)


def test_as_xyz_with_offset():
    """A non-zero offset shifts every point."""
    segs = {'seg1': WireSegment2D([(Real(0), Real(0)), (Real(10), Real(20))], 'segment one')}
    offset = np.array([1, 2, 3], dtype=Real)
    result = as_xyz(segs, _IDENTITY_FRAME, offset)
    pts = result['seg1']
    assert r3vector_equality(pts[0], np.array([1, 2, 3], dtype=Real), TOLERANCE)
    assert r3vector_equality(pts[1], np.array([11, 22, 3], dtype=Real), TOLERANCE)


def test_as_xyz_rotated_frame():
    """u=y, v=z, w=x: SVG (u,v) becomes global (0, u, v)."""
    # u=y-axis, v=z-axis, w=x-axis  (right-handed: y cross z = x)
    frame = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=Real)
    segs = {'r': WireSegment2D([(Real(1), Real(2)), (Real(3), Real(4))], 'rotated line')}
    result = as_xyz(segs, frame, _ZERO_OFFSET)
    pts = result['r']
    # point (u=1, v=2) -> 1*y_hat + 2*z_hat = (0, 1, 2)
    assert r3vector_equality(pts[0], np.array([0, 1, 2], dtype=Real), TOLERANCE)
    # point (u=3, v=4) -> 3*y_hat + 4*z_hat = (0, 3, 4)
    assert r3vector_equality(pts[1], np.array([0, 3, 4], dtype=Real), TOLERANCE)


def test_as_xyz_returns_real_dtype():
    segs = {
        'seg1': WireSegment2D([(Real(0), Real(0)), (Real(10), Real(20))], 'segment one'),
        'tri': WireSegment2D([(Real(0), Real(0)), (Real(5), Real(10)), (Real(10), Real(0))], 'triangle'),
    }
    result = as_xyz(segs, _IDENTITY_FRAME, _ZERO_OFFSET)
    for pts in result.values():
        for pt in pts:
            assert pt.dtype == Real


def test_as_xyz_empty_segments():
    result = as_xyz({}, _IDENTITY_FRAME, _ZERO_OFFSET)
    assert result == {}


def test_as_xyz_invalid_frame_raises():
    bad_frame = np.eye(3, dtype=Real) * 2  # not orthonormal
    with pytest.raises(Unrecoverable):
        as_xyz({}, bad_frame, _ZERO_OFFSET)


def test_as_xyz_invalid_offset_raises():
    bad_offset = np.array([1, 2], dtype=Real)  # wrong length
    with pytest.raises(Unrecoverable):
        as_xyz({}, _IDENTITY_FRAME, bad_offset)


# ---------------------------------------------------------------------------
# export_polylines tests (moved from tests/test_svg.py)
# ---------------------------------------------------------------------------


def _parse_vtk(path):
    """Parse a legacy VTK POLYDATA file; return (points, lines) where
    points is a list of (x,y,z) float tuples and lines is a list of
    lists of integer indices."""
    with open(path, encoding='ascii') as f:
        lines_raw = f.read().splitlines()
    points = []
    lines_out = []
    i = 0
    while i < len(lines_raw):
        tok = lines_raw[i].strip()
        if tok.upper().startswith('POINTS'):
            n = int(tok.split()[1])
            i += 1
            while len(points) < n * 3:
                for chunk in lines_raw[i].split():
                    points.append(float(chunk))
                i += 1
            points = [(points[j], points[j+1], points[j+2])
                      for j in range(0, len(points), 3)]
        elif tok.upper().startswith('LINES'):
            n_lines = int(tok.split()[1])
            i += 1
            for _ in range(n_lines):
                nums = list(map(int, lines_raw[i].split()))
                lines_out.append(nums[1:nums[0]+1])
                i += 1
        else:
            i += 1
    return points, lines_out


def _make_segments():
    """Build a simple as_xyz-style segments dict for export tests."""
    return {
        'a': [np.array([0, 0, 0], dtype=Real), np.array([1, 0, 0], dtype=Real)],
        'b': [np.array([0, 1, 0], dtype=Real), np.array([0, 1, 1], dtype=Real),
              np.array([0, 1, 2], dtype=Real)],
    }


def test_export_polylines_creates_file(tmp_path):
    out = str(tmp_path / 'out.vtk')
    export_polylines(_make_segments(), out)
    import os
    assert os.path.isfile(out)


def test_export_polylines_vtk_header(tmp_path):
    out = str(tmp_path / 'out.vtk')
    export_polylines(_make_segments(), out)
    with open(out, encoding='ascii') as f:
        header = f.readline().strip()
    assert header.startswith('# vtk DataFile Version')


def test_export_polylines_point_count(tmp_path):
    out = str(tmp_path / 'out.vtk')
    segs = _make_segments()
    export_polylines(segs, out)
    pts, _ = _parse_vtk(out)
    total = sum(len(v) for v in segs.values())
    assert len(pts) == total


def test_export_polylines_line_count(tmp_path):
    out = str(tmp_path / 'out.vtk')
    segs = _make_segments()
    export_polylines(segs, out)
    _, vtk_lines = _parse_vtk(out)
    assert len(vtk_lines) == len(segs)


def test_export_polylines_connectivity(tmp_path):
    out = str(tmp_path / 'out.vtk')
    segs = _make_segments()
    export_polylines(segs, out)
    pts, vtk_lines = _parse_vtk(out)
    # Reconstruct coordinates from connectivity and compare to input.
    all_input = [tuple(float(c) for c in pt)
                 for name in segs for pt in segs[name]]
    for line_indices in vtk_lines:
        for idx in line_indices:
            assert pts[idx] == all_input[idx]


def test_export_polylines_segment_lengths(tmp_path):
    out = str(tmp_path / 'out.vtk')
    segs = _make_segments()
    export_polylines(segs, out)
    _, vtk_lines = _parse_vtk(out)
    expected_lengths = [len(v) for v in segs.values()]
    assert [len(l) for l in vtk_lines] == expected_lengths


def test_export_polylines_coordinate_values(tmp_path):
    out = str(tmp_path / 'out.vtk')
    segs = {'seg': [np.array([1.5, 2.5, 3.5], dtype=Real),
                    np.array([4.5, 5.5, 6.5], dtype=Real)]}
    export_polylines(segs, out)
    pts, _ = _parse_vtk(out)
    assert pts[0] == (1.5, 2.5, 3.5)
    assert pts[1] == (4.5, 5.5, 6.5)


def test_export_polylines_empty_segments(tmp_path):
    out = str(tmp_path / 'out.vtk')
    export_polylines({}, out)
    pts, vtk_lines = _parse_vtk(out)
    assert pts == []
    assert vtk_lines == []


def test_export_polylines_bad_path_raises():
    with pytest.raises(Unrecoverable):
        export_polylines({}, '/nonexistent/directory/out.vtk')
