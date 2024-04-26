# Prerequisite packages
#
# All inputs will use CSV formatting for expedience
using CSV
# Enumerated data will be managed in DataFrame instances
using DataFrames
# Vector operations like norm
using LinearAlgebra
using Printf

# Given an IO, extract points under segments under "Start Stop Radius"  header, and append unique names.
# Names are based upon user-entered Start-Stop value pairs.
# Segments are also sorted after naming by lowest-to-highest in Start,Stop point indices.
function read_segments(points::DataFrame,segments::IO)::DataFrame
    parsed_segments = CSV.read(segments,DataFrame,comment="#",delim=" ")
    # Generate user-facing names
    labels = string.(parsed_segments.Start,"-",parsed_segments.Stop)
    # For internmal reference, order everything
    ordered_start = ifelse.(parsed_segments.Start .< parsed_segments.Stop, parsed_segments.Start,parsed_segments.Stop)
    ordered_stop = ifelse.(parsed_segments.Start .< parsed_segments.Stop, parsed_segments.Stop, parsed_segments.Start)
    ordered_segments = DataFrame(Name=labels,Start=ordered_start,Stop=ordered_stop,Radius=parsed_segments.Radius)
    sort!(ordered_segments,[:Start,:Stop])
    segment_count = nrow(ordered_segments)
    lengths = Vector{Float64}(undef,segment_count)
    counts = Vector{Int32}(undef,segment_count)
    start_points = Array{Float64}(undef,segment_count,3)
    stop_points = Array{Float64}(undef,segment_count,3)
    subsegment_density = 10.0
    for row = 1:segment_count
        start_index = ordered_segments.Start[row]
        start_points[row,:] = [points.X[start_index],points.Y[start_index],points.Z[start_index]]
        stop_index = ordered_segments.Stop[row]
        stop_points[row,:] = [points.X[stop_index],points.Y[stop_index],points.Z[stop_index]]
        points.Uses[start_index] += 1
        points.Uses[stop_index] += 1
        lengths[row] = norm(stop_points[row,:]-start_points[row,:])
        counts[row] = max(1,Int32(round(subsegment_density*lengths[row])))
    end
    segment_metadata = DataFrame(Name=ordered_segments.Name,Length=lengths,Count=counts)
    segments = innerjoin(ordered_segments,segment_metadata,on = :Name)
    # Note that start_points[row,:] and stop_points[row,:] now have 3D positions defined in them
    single_use_points = points[points.Uses .== 1, :]
    shared_points = points[points.Uses .> 1, :]
    if nrow(single_use_points) + nrow(shared_points) != nrow(points)
        throw(DomainError("Segments leave some points unused"))
    end
    return segments
end

# Subsegments are subdivisions of user-defined geometry segments
struct SubSegment
    start::Vector{Float64}
    stop::Vector{Float64}
    shared::Vector{Bool}
    SubSegment() = new(zeros(Float64,3),zeros(Float64,3),[false,false])
end

function get_subsegment!(points::DataFrame,segments::DataFrame,index::Int32,result::SubSegment)
    # Uses segments, start_points, stop_points
    search::Int32 = index
    row::Int32 = 1
    while row <= nrow(segments)
        if search <= segments.Count[row]
            # requested index is in this segment_count
            start_index = segments.Start[row]
            start_point = [points.X[start_index],points.Y[start_index],points.Z[start_index]]
            stop_index = segments.Stop[row]
            stop_point = [points.X[stop_index],points.Y[stop_index],points.Z[stop_index]]
            segment_delta = (stop_point-start_point)/segments.Count[row]
            result.start .= start_point + (search-1)*segment_delta
            result.stop .= start_point + search*segment_delta
            # detect endpoints not shared with other segments
            if search == 1
                result.shared[1] =(points.Uses[start_index] != 1)
            else
                result.shared[1] = true
            end
            if search == segments.Count[row]
                result.shared[2] = (points.Uses[stop_index] != 1)
            else
                result.shared[2] = true
            end
            break
        else
            search -= segments.Count[row]
            row += 1
        end
    end
    # Exiting loop by exceeding row count is a failure
    if row > nrow(segments)
        throw(DomainError("requested index not included in segments"))
    end
    return
end
