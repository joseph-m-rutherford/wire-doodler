# Prerequisite packages
#
# All inputs will use CSV formatting for expedience
using CSV
# Enumerated data will be managed in DataFrame instances
using DataFrames
using Printf

# Given an IO, extract points under X Y Z header and append unique names, use counts.
# Names are strings from index number (not line number).
# Adds zeroed integers for "Uses" to be updated by connectivity graph.
function read_points(points::IO)::DataFrame
    parsed_points = CSV.read(points,DataFrame,comment="#",delim=" ")
    point_count = nrow(parsed_points)
    point_names = Vector{String}(undef,point_count)
    for index = 1:point_count
        point_names[index] = @sprintf("%d",index)
    end
    named_points = DataFrame(Name=point_names,Uses=zeros(Int32,point_count),X=parsed_points.X,Y=parsed_points.Y,Z=parsed_points.Z)
    return named_points
end

function get_point!(points::DataFrame,index::Int32,result::Vector{Float64})
    result .= [points.X[index],points.Y[index],points.Z[index]]
end

function interior_shared_point_count(points::DataFrame,segments::DataFrame)
    # Uses segments, start_points, stop_points
    result::Int32 = 0
    # shared point index is total point count - unshared user points
    row::Int32 = 1
    while row <= nrow(segments)
        # Shared vertices exclusive to this segment is gotten by skipping user-defined end points
        result += (segments.Count[row] - 1)
        row += 1
    end
    return result
end

function get_interior_shared_point!(points::DataFrame,segments::DataFrame,index::Int32,result::Vector{Float64})
    # Uses segments, start_points, stop_points
    # shared point index is total point count - unshared user points
    row::Int32 = 1
    search::Int32 = index
    while row <= nrow(segments)
        # Shared vertices exclusive to this segment is gotten by skipping user-defined end points
        shared_point_count = segments.Count[row] - 1
        if search <= shared_point_count
            # requested index is in this segment_count
            start_index = segments.Start[row]
            start_point = [named_points.X[start_index],named_points.Y[start_index],named_points.Z[start_index]]
            stop_index = segments.Stop[row]
            stop_point = [named_points.X[stop_index],named_points.Y[stop_index],named_points.Z[stop_index]]
            segment_delta = (stop_point-start_point)/segments.Count[row]
            result .= start_point + search*segment_delta
            break
        else
            # Stride over this segment's interior shared points
            search -= shared_point_count
            row += 1
        end
    end
    # Exiting loop by exceeding row count is a failure
    if row > nrow(segments)
        throw(DomainError("requested index not included in segments"))
    end
end

function get_boundary_shared_point!(points::DataFrame,segments::DataFrame,index::Int32,result::Vector{Float64})
    # Boundary points are specified by user
    # Shared points have > 1 use in segments
    shared_points = points[points.Uses > 1,:]
    if index > nrow(shared_points)
        throw(DomainError("requested index exceeds shared user point count"))
    end
    result .= [shared_points.X[index],shared_points.Y[index],shared_points.Z[index]]
end

function get_boundary_unshared_point!(points::DataFrame,segments::DataFrame,index::Int32,result::Vector{Float64})
    # Boundary points are specified by user
    # Unshared points have > 1 use in segments
    shared_points = points[points.Uses .== 1,:]
    if index > nrow(shared_points)
        throw(DomainError("requested index exceeds shared user point count"))
    end
    result .= [shared_points.X[index],shared_points.Y[index],shared_points.Z[index]]
end