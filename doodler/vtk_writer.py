#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from .common import Real
from .errors import Unrecoverable
from .r3 import R3Vector


def export_polylines(segments: dict[str, list[R3Vector]], filename: str) -> None:
    """Write 3-D segment data to a legacy VTK ASCII file as LINES.

    Parameters
    ----------
    segments:
        Output of :func:`as_xyz` — a dict mapping names to lists of
        length-3 numpy arrays in global x, y, z coordinates.
    filename:
        Destination file path.  Overwrites existing files.

    Raises
    ------
    Unrecoverable:
        If the output file cannot be written.
    """
    # Flatten all points and record per-segment index ranges.
    all_points: list[R3Vector] = []
    line_index_lists: list[list[int]] = []
    for pts in segments.values():
        start = len(all_points)
        all_points.extend(pts)
        line_index_lists.append(list(range(start, start + len(pts))))

    n_points = len(all_points)
    n_lines = len(line_index_lists)
    # VTK legacy: total_size = sum of (npts_per_line + 1) for each line
    total_size = sum(len(idx) + 1 for idx in line_index_lists)

    try:
        with open(filename, 'w', encoding='ascii') as f:
            f.write('# vtk DataFile Version 2.0\n')
            f.write('wire-doodler polylines\n')
            f.write('ASCII\n')
            f.write('DATASET POLYDATA\n')
            f.write(f'POINTS {n_points} float\n')
            for pt in all_points:
                f.write(f'{float(pt[0])} {float(pt[1])} {float(pt[2])}\n')
            f.write(f'LINES {n_lines} {total_size}\n')
            for indices in line_index_lists:
                f.write(' '.join([str(len(indices))] + [str(i) for i in indices]) + '\n')
    except OSError as exc:
        raise Unrecoverable(exc)
