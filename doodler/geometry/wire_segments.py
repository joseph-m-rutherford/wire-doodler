#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from ..common import Real
from ..errors import NeverImplement


class WireSegment2D:
    """A single SVG path element parsed into 2-D geometry and a description.

    Attributes
    ----------
    points:
        Ordered list of (x, y) coordinate pairs parsed from the path ``d``
        attribute.
    description:
        Non-empty string taken from the required ``<desc>`` child element.
    """

    def __init__(self, points: list[tuple[Real, Real]], description: str) -> None:
        self._points = points
        self._description = description

    @property
    def points(self) -> list[tuple[Real, Real]]:
        '''Ordered list of (x, y) coordinate pairs parsed from the path d attribute.'''
        return self._points

    @points.setter
    def points(self, value) -> None:
        raise NeverImplement('WireSegment2D points are immutable')

    @property
    def description(self) -> str:
        '''Non-empty string taken from the required <desc> child element.'''
        return self._description

    @description.setter
    def description(self, value) -> None:
        raise NeverImplement('WireSegment2D description is immutable')
