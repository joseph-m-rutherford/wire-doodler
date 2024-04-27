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
points = Wires.read_points(IOBuffer(points_text))

segments_text = """
Start Stop Radius
# Top line of 1-2-3-4
1 2 0.001
2 3 0.002
3 4 0.003
# Lower left branch 5-6-2
5 6 0.001
6 2 0.002
# Lower right branch 3-7
3 7 0.003
"""
segments = Wires.read_segments(points,IOBuffer(segments_text))

RELATIVE_TOLERANCE = 1e-12
ABSOLUTE_TOLERANCE = 1e-6

# Compute the distance between two points in the same N-dimensional space
function distance(a::Vector{Float64},b::Vector{Float64})
    return norm(a-b)
end

# Compute the distance between two points in the same N-dimensional space
function relative_distance(a::Vector{Float64},b::Vector{Float64})
    simple_distance::Float64 = distance(a,b)
    greatest_radius::Float64 = max(norm(a),norm(b))
    result::Float64 = simple_distance
    if greatest_radius > ABSOLUTE_TOLERANCE
        result /= greatest_radius
    end
    return result
end

@testset "Evaluation Functions" begin
    @test abs(distance([1000.,2.,3.],[1000.,2.,4.]) - 1.0) < ABSOLUTE_TOLERANCE
    @test abs(relative_distance([1000.,2.,3.],[1000.,2.,4.]) - 1.0/sqrt(1000020.0)) < RELATIVE_TOLERANCE
end

@testset "Geometry Access" begin
    
    @test nrow(points) == 7
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
    # Confirm lesser,greater ordering, and subsegment counts
    @test segments.Name == ["1-2","2-3","6-2","3-4","3-7","5-6"]
    @test segments.Start == [1,2,2,3,3,5]
    @test segments.Stop == [2,3,6,4,7,6]
    @test segments.Radius == [0.001,0.002,0.002,0.003,0.003,0.001]
    @test segments.Count == Int32[10,10,10,10,14,10]
end

@testset "Mesh Access" begin
    subsegment = Wires.SubSegment()
    test_point::Vector{Float64} = [0,0,0]
    reference_point::Vector{Float64} = [0,0,0]
    # Unshared points 1, 4, 5, 7
    Wires.get_boundary_unshared_point!(points,Int32(2),test_point)
    reference_point = [4.0,1.0,1.0]
    @test relative_distance(test_point,reference_point) < RELATIVE_TOLERANCE
    # Shared points 2,3,6
    Wires.get_boundary_shared_point!(points,Int32(2),test_point)
    reference_point = [3.0,1.0,1.0]
    # First subsegment of first segment 1-2
    Wires.get_subsegment!(points,segments,Int32(1),subsegment)
    reference_point = [1,1,1]
    @test relative_distance(subsegment.start,reference_point) < RELATIVE_TOLERANCE
    reference_point = [1.1,1.0,1.0]
    @test relative_distance(subsegment.stop,reference_point) < RELATIVE_TOLERANCE
    @test subsegment.shared[1] == false
    @test subsegment.shared[2] == true
    # Last subsegment of first segment 1-2
    Wires.get_subsegment!(points,segments,Int32(10),subsegment)
    reference_point = [1.9,1,1]
    @test relative_distance(subsegment.start,reference_point) < RELATIVE_TOLERANCE
    reference_point = [2.0,1.0,1.0]
    @test relative_distance(subsegment.stop,reference_point) < RELATIVE_TOLERANCE
    @test subsegment.shared[1] == true
    @test subsegment.shared[2] == true
    # First subsegment of third segment 2-6
    Wires.get_subsegment!(points,segments,Int32(21),subsegment)
    reference_point = [2.0,1,1]
    @test relative_distance(subsegment.start,reference_point) < RELATIVE_TOLERANCE
    reference_point = [2.0,0.9,1.0]
    @test relative_distance(subsegment.stop,reference_point) < RELATIVE_TOLERANCE
    @test subsegment.shared[1] == true
    @test subsegment.shared[2] == true
    # First subsegment of sixth segment 5-6
    Wires.get_subsegment!(points,segments,Int32(55),subsegment)
    reference_point = [1.0,0,1]
    @test relative_distance(subsegment.start,reference_point) < RELATIVE_TOLERANCE
    reference_point = [1.1,0,1]
    @test relative_distance(subsegment.stop,reference_point) < RELATIVE_TOLERANCE
    @test subsegment.shared[1] == false
    @test subsegment.shared[2] == true
    # Test subsegment point requests by index, first point in 1-2
    @test Wires.interior_shared_point_count(points,segments) == 9*5+13
    Wires.get_interior_shared_point!(points,segments,Int32(1),test_point)
    reference_point = [1.1,1.0,1.0]
    @test relative_distance(test_point,reference_point) < RELATIVE_TOLERANCE
    # Test subsegment point requests by index, last point in 5-6
    Wires.get_interior_shared_point!(points,segments,Int32(58),test_point)
    reference_point = [1.9,0.0,1.0]
    @test relative_distance(test_point,reference_point) < RELATIVE_TOLERANCE
end

# Close top-level testset
end