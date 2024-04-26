# Copyright (c) 2024, Joseph M. Rutherford
# All rights reserved.
#
# Code provided under the license contained in the LICENSE file.
#
# This file include the WireDoodlerTester module's source code, runs the tests, and writes a report.

include("../doodler/geometry/wires.jl")

using Test
using DataFrames
using LinearAlgebra
@testset verbose=true "Wire Doodler: geometry/wires" begin

    points_text = """
X Y Z
# Point p1
1.0 1.0 1.0
# Point p2
2.0 1.0 1.0
# Point p3
3.0 1.0 1.0
# Point p4
4.0 1.0 1.0
# point p5
1.0 0.0 1.0
# point p6
2.0 0.0 1.0
# point p7
4.0 0.0 1.0
"""

segments_text = """
Start Stop Radius
# Top line of 1-2-3-4
1 2 0.001
2 3 0.001
3 4 0.001
# Lower left branch 5-6-2
5 6 0.001
6 2 0.001
# Lower right branch 3-7
3 7 0.001
"""

RELATIVE_TOLERANCE = 1e-12
ABSOLUTE_TOLERANCE = 1e-6

function distance(a::Vector{Float64},b::Vector{Float64})
    return norm(a-b)
end

function relative_distance(a::Vector{Float64},b::Vector{Float64})
    return norm(a-b)/(norm(a)+norm(b))
end

@testset "Geometry Access" begin
    points = Wires.read_points(IOBuffer(points_text))
    @test nrow(points) == 7
    segments = Wires.read_segments(points,IOBuffer(segments_text))
    @test nrow(segments) == 6
    reference_points = [
        [1.0,1.0,1.0],
        [2.0,1.0,1.0],
        [3.0,1.0,1.0],
        [4.0,1.0,1.0],
        [1.0,0.0,1.0],
        [2.0,0.0,1.0],
        [4.0,0.0,1.0] ]
    test_point::Vector{Float64} = [0.,0.,0.]
    for i = 1:7
        Wires.get_point!(points,Int32(i),test_point)
        @test relative_distance(test_point,reference_points[i]) < RELATIVE_TOLERANCE
    end
    reference_use_counts::Vector{Int32} = [1,3,3,1,1,2,1]
    @test points.Uses == reference_use_counts
end

@testset "Mesh Access" begin
    points = Wires.read_points(IOBuffer(points_text))
    segments = Wires.read_segments(points,IOBuffer(segments_text))
    subsegment = Wires.SubSegment()
    Wires.get_subsegment!(points,segments,Int32(1),subsegment)
    reference_point::Vector{Float64} = [1,1,1]
    @test relative_distance(subsegment.start,reference_point) < RELATIVE_TOLERANCE
    reference_point = [1.1,1.0,1.0]
    @test relative_distance(subsegment.stop,reference_point) < RELATIVE_TOLERANCE
    @test subsegment.shared[1] == false
    @test subsegment.shared[2] == true
end

# Close top-level testset
end