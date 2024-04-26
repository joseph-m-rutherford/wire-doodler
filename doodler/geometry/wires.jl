# Copyright (c) 2024, Joseph M. Rutherford
# All rights reserved.
#
# Code provided under the license contained in the LICENSE file.
#
# This file defines outward-facing interfaces for "wires"
module Wires

include("points.jl")
export read_points, interior_shared_point_count, get_interior_shared_point!, get_boundary_shared_point!, get_boundary_unshared_point!

include("segments.jl")
export read_segments, get_subsegment!, SubSegment 

end