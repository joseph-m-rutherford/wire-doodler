#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import pytest

import numpy as np
import pytest

from doodler import read_svg, as_xyz, export_polylines, Real, Unrecoverable, NotYetImplemented, WireSegment2D, R3Axes, R3Vector, r3vector_equality, real_equality
from doodler.r3 import TOLERANCE


# ---------------------------------------------------------------------------
# Sample SVG strings
# ---------------------------------------------------------------------------

_SVG_NAMESPACED = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <line id="seg1" x1="0" y1="0" x2="10" y2="20"><desc>segment one</desc></line>
  <polyline id="tri" points="0,0 5,10 10,0"><desc>triangle</desc></polyline>
</svg>'''

_SVG_NO_NAMESPACE = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100">
  <line id="seg1" x1="1.5" y1="2.5" x2="3.5" y2="4.5"><desc>segment one</desc></line>
  <polyline id="path1" points="0 0 1 1 2 0"><desc>path one</desc></polyline>
</svg>'''

_SVG_COMMAS_AND_SPACES = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <polyline id="mixed" points="0,0, 3,4, 6,8"><desc>mixed separators</desc></polyline>
</svg>'''

_SVG_MISSING_ID = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <line x1="0" y1="0" x2="1" y2="1"/>
</svg>'''

_SVG_DUPLICATE_ID = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <line id="same" x1="0" y1="0" x2="1" y2="1"><desc>first</desc></line>
  <polyline id="same" points="0,0 1,1"><desc>second</desc></polyline>
</svg>'''

_SVG_EMPTY_POINTS = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <polyline id="empty" points=""/>
</svg>'''

_SVG_ODD_TOKENS = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <polyline id="bad" points="0 1 2"/>
</svg>'''

_SVG_DEFAULT_COORDS = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <line id="origin"/>
</svg>'''

_SVG_LINE_MISSING_DESC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <line id="seg1" x1="0" y1="0" x2="1" y2="1"/>
</svg>'''

_SVG_POLYLINE_MISSING_DESC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <polyline id="poly1" points="0,0 1,1 2,0"/>
</svg>'''

_SVG_LINE_EMPTY_DESC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <line id="seg1" x1="0" y1="0" x2="1" y2="1"><desc></desc></line>
</svg>'''

_SVG_POLYLINE_EMPTY_DESC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <polyline id="poly1" points="0,0 1,1 2,0"><desc></desc></polyline>
</svg>'''


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _svg_file(tmp_path, content, name='test.svg'):
    p = tmp_path / name
    p.write_text(content, encoding='utf-8')
    return str(p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_line_two_points(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NAMESPACED))
    assert 'seg1' in result
    pts = result['seg1'].points
    assert len(pts) == 2
    assert pts[0] == (Real(0), Real(0))
    assert pts[1] == (Real(10), Real(20))


def test_polyline_three_points(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NAMESPACED))
    assert 'tri' in result
    pts = result['tri'].points
    assert len(pts) == 3
    assert pts[0] == (Real(0), Real(0))
    assert pts[1] == (Real(5), Real(10))
    assert pts[2] == (Real(10), Real(0))


def test_both_elements_keyed(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NAMESPACED))
    assert set(result.keys()) == {'seg1', 'tri'}


def test_no_namespace(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NO_NAMESPACE))
    assert set(result.keys()) == {'seg1', 'path1'}
    assert len(result['path1'].points) == 3


def test_no_namespace_line_coords(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NO_NAMESPACE))
    pts = result['seg1'].points
    assert pts[0] == (Real('1.5'), Real('2.5'))
    assert pts[1] == (Real('3.5'), Real('4.5'))


def test_polyline_mixed_separators(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_COMMAS_AND_SPACES))
    pts = result['mixed'].points
    assert len(pts) == 3
    assert pts[1] == (Real(3), Real(4))


def test_missing_id_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_MISSING_ID))


def test_duplicate_id_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_DUPLICATE_ID))


def test_empty_points_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_EMPTY_POINTS))


def test_odd_token_count_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_ODD_TOKENS))


def test_missing_file_raises():
    with pytest.raises(Unrecoverable):
        read_svg('/nonexistent/no_such_file.svg')


def test_line_missing_coords_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_DEFAULT_COORDS))


def test_line_missing_desc_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_LINE_MISSING_DESC))


def test_polyline_missing_desc_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_POLYLINE_MISSING_DESC))


def test_line_empty_desc_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_LINE_EMPTY_DESC))


def test_polyline_empty_desc_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_POLYLINE_EMPTY_DESC))


def test_line_description_stored(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NAMESPACED))
    assert result['seg1'].description == 'segment one'


def test_polyline_description_stored(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NAMESPACED))
    assert result['tri'].description == 'triangle'


def test_coordinate_dtype(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NAMESPACED))
    for segment in result.values():
        for x, y in segment.points:
            assert type(x) is Real
            assert type(y) is Real


def test_returns_empty_dict_for_svg_with_no_lines(tmp_path):
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect id="r" width="10" height="10"/></svg>'
    result = read_svg(_svg_file(tmp_path, svg))
    assert result == {}


# ---------------------------------------------------------------------------
# as_xyz tests
# ---------------------------------------------------------------------------

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
# export_polylines tests
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


# ---------------------------------------------------------------------------
# path element tests
# ---------------------------------------------------------------------------

_SVG_PATH_LINE = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="seg" d="M 0 0 L 10 20">
    <desc>wire segment</desc>
  </path>
</svg>'''

_SVG_PATH_POLYLINE = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="tri" d="M 0 0 L 5 10 L 10 0">
    <desc>triangle path</desc>
  </path>
</svg>'''

_SVG_PATH_RELATIVE = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="rel" d="m 1 1 l 2 0 l 0 2">
    <desc>relative path</desc>
  </path>
</svg>'''

_SVG_PATH_HORIZONTAL_VERTICAL = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="hv" d="M 0 0 H 5 V 3">
    <desc>horizontal vertical path</desc>
  </path>
</svg>'''

_SVG_PATH_RELATIVE_HV = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="rhv" d="M 1 1 h 4 v 2">
    <desc>relative hv path</desc>
  </path>
</svg>'''

_SVG_PATH_CLOSE = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="sq" d="M 0 0 H 10 V 10 Z">
    <desc>closed path</desc>
  </path>
</svg>'''

_SVG_PATH_IMPLICIT_LINETO = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="imp" d="M 0 0 5 10 10 0">
    <desc>implicit lineto</desc>
  </path>
</svg>'''

_SVG_PATH_ARC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="arc" d="M 0 0 A 5 5 0 0 1 10 0">
    <desc>arc path</desc>
  </path>
</svg>'''

_SVG_PATH_CUBIC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="cub" d="M 0 0 C 1 1 2 2 3 0">
    <desc>cubic bezier</desc>
  </path>
</svg>'''

_SVG_PATH_QUADRATIC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="qua" d="M 0 0 Q 5 10 10 0">
    <desc>quadratic bezier</desc>
  </path>
</svg>'''

_SVG_PATH_SMOOTH_CUBIC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="sc" d="M 0 0 S 5 5 10 0">
    <desc>smooth cubic bezier</desc>
  </path>
</svg>'''

_SVG_PATH_SMOOTH_QUADRATIC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="sq2" d="M 0 0 T 10 0">
    <desc>smooth quadratic bezier</desc>
  </path>
</svg>'''

_SVG_PATH_MISSING_ID = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path d="M 0 0 L 1 1"/>
</svg>'''

_SVG_PATH_EMPTY_D = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="empty" d="">
    <desc>empty d path</desc>
  </path>
</svg>'''

_SVG_PATH_DUPLICATE_ID = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="same" d="M 0 0 L 1 1">
    <desc>first path</desc>
  </path>
  <path id="same" d="M 2 2 L 3 3">
    <desc>second path</desc>
  </path>
</svg>'''

_SVG_MIXED_ELEMENTS = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <line id="ln" x1="0" y1="0" x2="1" y2="1"><desc>mixed line</desc></line>
  <polyline id="pl" points="0,0 2,0 2,2"><desc>mixed polyline</desc></polyline>
  <path id="pa" d="M 3 0 L 5 5">
    <desc>mixed path element</desc>
  </path>
</svg>'''


def test_path_line_two_points(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_LINE))
    assert 'seg' in result
    pts = result['seg'].points
    assert len(pts) == 2
    assert pts[0] == (Real(0), Real(0))
    assert pts[1] == (Real(10), Real(20))


def test_path_polyline_three_points(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_POLYLINE))
    pts = result['tri'].points
    assert len(pts) == 3
    assert pts[0] == (Real(0), Real(0))
    assert pts[1] == (Real(5), Real(10))
    assert pts[2] == (Real(10), Real(0))


def test_path_relative_commands(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_RELATIVE))
    pts = result['rel'].points
    assert len(pts) == 3
    assert pts[0] == (Real(1), Real(1))
    assert pts[1] == (Real(3), Real(1))
    assert pts[2] == (Real(3), Real(3))


def test_path_absolute_horizontal_vertical(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_HORIZONTAL_VERTICAL))
    pts = result['hv'].points
    assert len(pts) == 3
    assert pts[0] == (Real(0), Real(0))
    assert pts[1] == (Real(5), Real(0))
    assert pts[2] == (Real(5), Real(3))


def test_path_relative_horizontal_vertical(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_RELATIVE_HV))
    pts = result['rhv'].points
    assert len(pts) == 3
    assert pts[0] == (Real(1), Real(1))
    assert pts[1] == (Real(5), Real(1))
    assert pts[2] == (Real(5), Real(3))


def test_path_close_appends_start_point(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_CLOSE))
    pts = result['sq'].points
    assert len(pts) == 4
    assert pts[0] == (Real(0), Real(0))
    assert pts[-1] == (Real(0), Real(0))


def test_path_implicit_lineto_after_move(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_IMPLICIT_LINETO))
    pts = result['imp'].points
    assert len(pts) == 3
    assert pts[0] == (Real(0), Real(0))
    assert pts[1] == (Real(5), Real(10))
    assert pts[2] == (Real(10), Real(0))


def test_path_arc_raises_not_yet_implemented(tmp_path):
    with pytest.raises(NotYetImplemented):
        read_svg(_svg_file(tmp_path, _SVG_PATH_ARC))


def test_path_cubic_bezier_raises_not_yet_implemented(tmp_path):
    with pytest.raises(NotYetImplemented):
        read_svg(_svg_file(tmp_path, _SVG_PATH_CUBIC))


def test_path_quadratic_bezier_raises_not_yet_implemented(tmp_path):
    with pytest.raises(NotYetImplemented):
        read_svg(_svg_file(tmp_path, _SVG_PATH_QUADRATIC))


def test_path_smooth_cubic_raises_not_yet_implemented(tmp_path):
    with pytest.raises(NotYetImplemented):
        read_svg(_svg_file(tmp_path, _SVG_PATH_SMOOTH_CUBIC))


def test_path_smooth_quadratic_raises_not_yet_implemented(tmp_path):
    with pytest.raises(NotYetImplemented):
        read_svg(_svg_file(tmp_path, _SVG_PATH_SMOOTH_QUADRATIC))


def test_path_missing_id_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_PATH_MISSING_ID))


def test_path_empty_d_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_PATH_EMPTY_D))


def test_path_duplicate_id_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_PATH_DUPLICATE_ID))


def test_path_coordinate_dtype(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_POLYLINE))
    for wire_seg in result.values():
        for x, y in wire_seg.points:
            assert type(x) is Real
            assert type(y) is Real


def test_mixed_line_polyline_path(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_MIXED_ELEMENTS))
    assert set(result.keys()) == {'ln', 'pl', 'pa'}
    assert len(result['ln'].points) == 2
    assert len(result['pl'].points) == 3
    assert len(result['pa'].points) == 2


# ---------------------------------------------------------------------------
# WireSegment2D and path <desc> tests
# ---------------------------------------------------------------------------

_SVG_PATH_WITH_DESC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="w" d="M 1 2 L 3 4">
    <desc>material=copper radius=0.5</desc>
  </path>
</svg>'''

_SVG_PATH_MISSING_DESC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="nd" d="M 0 0 L 1 1"/>
</svg>'''

_SVG_PATH_EMPTY_DESC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="ed" d="M 0 0 L 1 1">
    <desc></desc>
  </path>
</svg>'''

_SVG_PATH_WHITESPACE_DESC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path id="wd" d="M 0 0 L 1 1">
    <desc>   </desc>
  </path>
</svg>'''

_SVG_PATH_NO_NAMESPACE_DESC = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg>
  <path id="nn" d="M 0 0 L 5 5">
    <desc>no namespace path</desc>
  </path>
</svg>'''


def test_path_returns_wire_segment_2d(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_WITH_DESC))
    assert isinstance(result['w'], WireSegment2D)


def test_path_description_stored(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_WITH_DESC))
    assert result['w'].description == 'material=copper radius=0.5'


def test_path_points_accessible_via_wire_segment(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_WITH_DESC))
    pts = result['w'].points
    assert pts[0] == (Real(1), Real(2))
    assert pts[1] == (Real(3), Real(4))


def test_path_missing_desc_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_PATH_MISSING_DESC))


def test_path_empty_desc_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_PATH_EMPTY_DESC))


def test_path_whitespace_desc_raises(tmp_path):
    with pytest.raises(Unrecoverable):
        read_svg(_svg_file(tmp_path, _SVG_PATH_WHITESPACE_DESC))


def test_path_desc_no_namespace(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_NO_NAMESPACE_DESC))
    assert isinstance(result['nn'], WireSegment2D)
    assert result['nn'].description == 'no namespace path'


def test_wire_segment_2d_has_points_attribute(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_WITH_DESC))
    assert hasattr(result['w'], 'points')


def test_wire_segment_2d_has_description_attribute(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_PATH_WITH_DESC))
    assert hasattr(result['w'], 'description')



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
