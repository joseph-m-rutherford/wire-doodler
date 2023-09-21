#! julia
import Arrow
import DataFrames
import Parquet
import QuadGK

function gauss_kronrod(n::Integer)
    # Get minimal quadrature rule data from QuadGK
    values, k_weights, g_weights = QuadGK.kronrod(n)
    # Expand values and k_weights about zero value
    full_k_x = vcat(values,-1*reverse(values)[2:length(values)])
    full_k_weights = vcat(k_weights,reverse(k_weights)[2:length(k_weights)])
    kronrod_result = DataFrames.DataFrame(order=(2*n+1), position=full_k_x, weight=full_k_weights)
    # Now compute complimentary Gauss rule
    half_gauss_values = values[2:2:n+1]
    if (n % 2) == 0
        # Even Gauss order
        full_g_x = vcat(half_gauss_values, -1*reverse(half_gauss_values))
        full_g_weights = vcat(g_weights, reverse(g_weights))
    else
        # Odd Gauss order
        full_g_x = vcat(half_gauss_values, -1*reverse(half_gauss_values)[2:length(half_gauss_values)])
        full_g_weights = vcat(g_weights, reverse(g_weights)[2:length(half_gauss_values)])
    end
    gauss_result = DataFrames.DataFrame(order=n, position=full_g_x, weight=full_g_weights)
    # Gauss rule and Kronod rule are returned in separate tables
    return gauss_result, kronrod_result
end
# Compute quadrature rules and export to disk
@Threads.threads for order = 5:51
    k_order = 2*order+1
    gauss_table, kronrod_table = gauss_kronrod(order)
    open("gauss_$order.parquet",write=true) do out
        Parquet.write_parquet(out,gauss_table)
    end
    open("kronrod_$k_order.parquet",write=true) do out
        Parquet.write_parquet(out,kronrod_table)
    end
end