# Variational Distributions

module VariationalDistributions
using DistributionsAD
using StatsFuns


function meanfield(z::AbstractArray{Float32}; tot_params::Int64)
    DistributionsAD.MultivariateNormal(
        z[1:tot_params],
        StatsFuns.softplus.(z[(tot_params + 1):(tot_params * 2)])
    )
end

end
