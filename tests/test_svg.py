#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import pytest

from doodler import read_svg, Real, Unrecoverable


# ---------------------------------------------------------------------------
# Sample SVG strings
# ---------------------------------------------------------------------------

_SVG_NAMESPACED = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <line id="seg1" x1="0" y1="0" x2="10" y2="20"/>
  <polyline id="tri" points="0,0 5,10 10,0"/>
</svg>'''

_SVG_NO_NAMESPACE = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100">
  <line id="seg1" x1="1.5" y1="2.5" x2="3.5" y2="4.5"/>
  <polyline id="path1" points="0 0 1 1 2 0"/>
</svg>'''

_SVG_COMMAS_AND_SPACES = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <polyline id="mixed" points="0,0, 3,4, 6,8"/>
</svg>'''

_SVG_MISSING_ID = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <line x1="0" y1="0" x2="1" y2="1"/>
</svg>'''

_SVG_DUPLICATE_ID = '''\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <line id="same" x1="0" y1="0" x2="1" y2="1"/>
  <polyline id="same" points="0,0 1,1"/>
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
    pts = result['seg1']
    assert len(pts) == 2
    assert pts[0] == (Real(0), Real(0))
    assert pts[1] == (Real(10), Real(20))


def test_polyline_three_points(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NAMESPACED))
    assert 'tri' in result
    pts = result['tri']
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
    assert len(result['path1']) == 3


def test_no_namespace_line_coords(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NO_NAMESPACE))
    pts = result['seg1']
    assert pts[0] == (Real('1.5'), Real('2.5'))
    assert pts[1] == (Real('3.5'), Real('4.5'))


def test_polyline_mixed_separators(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_COMMAS_AND_SPACES))
    pts = result['mixed']
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


def test_line_missing_coords_defaults_to_zero(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_DEFAULT_COORDS))
    pts = result['origin']
    assert pts[0] == (Real(0), Real(0))
    assert pts[1] == (Real(0), Real(0))


def test_coordinate_dtype(tmp_path):
    result = read_svg(_svg_file(tmp_path, _SVG_NAMESPACED))
    for pts in result.values():
        for x, y in pts:
            assert type(x) is Real
            assert type(y) is Real


def test_returns_empty_dict_for_svg_with_no_lines(tmp_path):
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect id="r" width="10" height="10"/></svg>'
    result = read_svg(_svg_file(tmp_path, svg))
    assert result == {}
