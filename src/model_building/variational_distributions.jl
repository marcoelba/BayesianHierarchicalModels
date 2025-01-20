# Variational Distributions

module VariationalDistributions
using DistributionsAD
using Bijectors
using StatsFuns
using LogExpFunctions


function get_variational_dist(z::AbstractArray, q_family_array::AbstractArray, range_z::AbstractVector)
    q_vi = [q_family_array[cc](z[range_z[cc]]) for cc in eachindex(q_family_array)]
    return q_vi
end


function rand_array(q_dist_array::AbstractArray; from_base_dist::Bool=false, reduce_to_vec::Bool=false)
    if from_base_dist
        v_sample = [rand(dist.dist) for dist in q_dist_array]
    else
        v_sample = [rand(dist) for dist in q_dist_array]
    end

    if reduce_to_vec
        v_sample = reduce(vcat, v_sample)
    end

    return v_sample
end


function rand_with_logjacobian(q_dist_array::AbstractArray)
    x = rand_array(q_dist_array, from_base_dist=true, reduce_to_vec=false)
    x_t = [q_dist_array[ii].transform(x[ii]) for ii in eachindex(x)]
    abs_jacobian = [Bijectors.jacobian(q_dist_array[ii].transform, x[ii], x_t[ii]) for ii in eachindex(x)]

    return x_t, sum(abs_jacobian)
end


# Entropy
function entropy(d::Bijectors.TransformedDistribution)
    DistributionsAD.entropy(d.dist)
end


function meanfield(z::AbstractArray{Float32}; tot_params::Int64)
    DistributionsAD.MultivariateNormal(
        z[1:tot_params],
        LogExpFunctions.softplus.(z[(tot_params + 1):(tot_params * 2)])
    )
end


function vi_mv_normal(z::AbstractArray; bij=identity)
    zdim = length(z)
    pdim = Int(zdim / 2)

    Bijectors.transformed(
        DistributionsAD.MvNormal(
            z[1:pdim],
            LogExpFunctions.log1pexp.(z[pdim+1:zdim])
        ),
        bij
    )
end


function vi_normal(z::AbstractArray; bij=identity)
    Bijectors.transformed(
        DistributionsAD.Normal(
            z[1],
            LogExpFunctions.log1pexp.(z[2])
        ),
        bij
    )
end


end
